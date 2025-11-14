# app.py

# -*- coding: utf-8 -*-

"""
Simple KV Proxy:

- Большие: LCP→restore на свободный/старый слот, затем чат строго в этот же слот, потом save+meta.
- Малые: свободный/старый слот, без restore и без дискового save/meta.
- Пин slota дублируется в root/options/query (через клиента).
- Ключи кеша и мета завязаны на model_id, полученный от llama.cpp,
  но /v1/models прокси по-прежнему отдаёт MODEL_ID из конфигурации.
"""

import asyncio
import time
import logging
from typing import List, Dict, AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from config import BACKENDS, BIG_THRESHOLD_WORDS, LCP_TH, MODEL_ID, WORDS_PER_BLOCK, PORT
import hashing as hs
from llama_client import LlamaClient
from slot_manager import SlotManager, GSlot

log = logging.getLogger(__name__)

app = FastAPI(title="Simple KV Proxy")


@app.on_event("startup")
async def startup():
    clients = [LlamaClient(be["url"]) for be in BACKENDS]
    sm = SlotManager()
    sm.set_clients(clients)
    app.state.clients = clients
    app.state.sm = sm
    log.info("app_start n_backends=%d port=%d", len(BACKENDS), PORT)


@app.on_event("shutdown")
async def shutdown():
    clients: List[LlamaClient] = getattr(app.state, "clients", [])
    if clients:
        await asyncio.gather(*(c.close() for c in clients))


@app.get("/v1/models")
async def models():
    # Внешний API: всегда отдаём MODEL_ID, как и раньше
    return {"data": [{"id": MODEL_ID}]}


async def stream_bytes_gen(
    resp: httpx.Response,
    g: GSlot,
    key: str,
    prefix: str,
    blocks: List[str],
    model_id: str,
    sm: SlotManager,
    is_big: bool,
) -> AsyncGenerator[bytes, None]:
    try:
        async for chunk in resp.aiter_raw():
            if chunk:
                yield chunk
    finally:
        try:
            await resp.aclose()
        except Exception:
            pass

        # После завершения стрима сохраняем только большие запросы
        ok = await sm.save_after(g, key, is_big=is_big)
        if is_big:
            hs.write_meta(key, prefix, blocks, WORDS_PER_BLOCK, model_id)
        sm.release(g)
        log.info(
            "stream_done g=%s key=%s saved=%s is_big=%s",
            g,
            key[:16],
            ok,
            is_big,
        )


@app.post("/v1/chat/completions")
async def chat(req: Request):
    sm: SlotManager = app.state.sm
    clients: List[LlamaClient] = app.state.clients

    t0 = time.time()
    data = await req.json()

    messages: List[Dict] = data.get("messages") or []
    stream = bool(data.get("stream", False))
    client_model = data.get("model") or MODEL_ID

    # model_id для кеша берём у первого backend'а (предполагается, что все backends одинаковые)
    # Это id из самого llama.cpp, а не клиентское имя модели.
    backend_model_id = await clients[0].get_model_id()

    prefix = hs.raw_prefix(messages)
    # Ключ завязан и на модель, и на префикс
    full_for_key = backend_model_id + "\n" + prefix
    key = hs.prefix_key_sha256(full_for_key)

    blocks = hs.block_hashes_from_text(prefix, WORDS_PER_BLOCK)
    n_words = len(hs.words_from_text(prefix))
    is_big = n_words > BIG_THRESHOLD_WORDS

    restore_key: Optional[str] = None
    if is_big:
        cand = hs.find_best_restore_candidate(
            blocks,
            WORDS_PER_BLOCK,
            LCP_TH,
            backend_model_id,
        )
        if cand:
            restore_key, ratio = cand
            log.info(
                "restore_candidate basename=%s ratio=%.3f",
                restore_key[:16],
                ratio,
            )
        else:
            log.info("restore_candidate none")
    else:
        log.info(
            "small_request n_words=%d threshold=%d",
            n_words,
            BIG_THRESHOLD_WORDS,
        )

    # Получаем слот: логика выбора + обязательный save при эвикте для больших внутри SlotManager
    # (предполагается обновлённый SlotManager.acquire_for_request(is_big, restore_key))
    g, lock, restored = await sm.acquire_for_request(
        is_big=is_big,
        restore_key=restore_key if is_big else None,
    )

    be_id, slot_id = g
    client = clients[be_id]

    # Формируем тело: корневые флаги для cache_prompt и n_keep
    body = dict(data)
    # Наружу используем клиентскую модель или MODEL_ID, как и раньше
    body["model"] = client_model
    body["cache_prompt"] = bool(is_big)
    body["n_keep"] = -1

    opts = dict(body.get("options") or {})
    opts["slot_id"] = slot_id
    opts["id_slot"] = slot_id
    opts["n_keep"] = -1
    opts["cache_prompt"] = bool(is_big)
    body["options"] = opts

    log.info(
        "dispatch be=%d slot=%d is_big=%s (restore_target=%s restored=%s model_id=%s)",
        be_id,
        slot_id,
        is_big,
        restore_key[:16] if restore_key else None,
        restored,
        backend_model_id,
    )

    try:
        if stream:
            resp = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=True,
            )
            if resp.status_code != 200:
                err_txt = await resp.aread()
                await resp.aclose()
                sm.release(g)
                return JSONResponse(
                    {"error": err_txt.decode("utf-8", "ignore")},
                    status_code=resp.status_code,
                )

            async def gen():
                async for chunk in stream_bytes_gen(
                    resp,
                    g,
                    key,
                    prefix,
                    blocks,
                    backend_model_id,
                    sm,
                    is_big=is_big,
                ):
                    yield chunk

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers=headers,
            )

        else:
            out = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=False,
            )
            if not isinstance(out, dict):
                sm.release(g)
                return JSONResponse(
                    {"error": "provider non-JSON body"},
                    status_code=502,
                )

            # После нестримового ответа сохраняем только большие запросы
            ok = await sm.save_after(g, key, is_big=is_big)
            if is_big:
                hs.write_meta(
                    key,
                    prefix,
                    blocks,
                    WORDS_PER_BLOCK,
                    backend_model_id,
                )
            sm.release(g)
            log.info(
                "json_done g=%s key=%s saved=%s is_big=%s dur_ms=%d",
                g,
                key[:16],
                ok,
                is_big,
                int((time.time() - t0) * 1000),
            )
            return JSONResponse(content=out, status_code=200)

    except Exception as e:
        sm.release(g)
        log.exception("chat_error g=%s key=%s: %s", g, key[:16], e)
        return JSONResponse({"error": str(e)}, status_code=500)
