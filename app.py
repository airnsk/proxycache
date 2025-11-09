# app.py
# -*- coding: utf-8 -*-
import time
import httpx
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Tuple, List, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from config import (
    BACKENDS,
    DEFAULT_MODEL_ID,
    DEFAULT_WORDS_PER_BLOCK,
    THRESHOLD_MODE,
    MIN_PREFIX_CHARS,
    MIN_PREFIX_WORDS,
    MIN_PREFIX_BLOCKS,
)
from llama_client import LlamaClient
from slot_manager import SlotManager, Backend
from hashing import (
    canonical_chat_prefix,
    block_hashes_from_text,
    prefix_key_sha256,
    words_from_text,
)

log = logging.getLogger("proxycache")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Создаём клиентов для всех бэкендов
    backends: List[Backend] = []
    for idx, desc in enumerate(BACKENDS):
        url = desc["url"]
        slots = int(desc["slots"])
        backends.append(Backend(id=idx, url=url, slots=slots, client=LlamaClient(url)))
        log.info(f"backend_registered id={idx} url={url} slots={slots}")
    app.state.backends = backends
    app.state.slot_manager = SlotManager(backends, DEFAULT_MODEL_ID, logger=log)
    log.info(f"lifespan_startup backends={len(backends)} total_slots={sum(b.slots for b in backends)}")
    try:
        yield
    finally:
        # Корректно закрываем клиентов
        for b in app.state.backends:
            await b.client.close()
        log.info("lifespan_shutdown")

app = FastAPI(title="OpenAI-compatible Llama Proxy (multi-backend global slot pool)", lifespan=lifespan)

def build_done_chunk() -> bytes:
    return b"data: [DONE]\n\n"

def resolve_block_size(request: Request) -> int:
    v = request.headers.get("x-block-size") or request.query_params.get("block_size")
    if v:
        try:
            n = int(v)
            if 1 <= n <= 2048:
                return n
        except Exception:
            pass
    return DEFAULT_WORDS_PER_BLOCK

def resolve_threshold_mode(request: Request) -> str:
    v = (request.headers.get("x-threshold-mode") or request.query_params.get("threshold_mode") or THRESHOLD_MODE).lower()
    return v if v in ("chars","words","blocks") else "chars"

def resolve_min_prefix_chars(request: Request) -> int:
    v = request.headers.get("x-min-prefix-chars") or request.query_params.get("min_prefix_chars")
    if v:
        try:
            n = int(v)
            if 0 <= n <= 10_000_000:
                return n
        except Exception:
            pass
    return MIN_PREFIX_CHARS

def resolve_min_prefix_words(request: Request) -> int:
    v = request.headers.get("x-min-prefix-words") or request.query_params.get("min_prefix_words")
    if v:
        try:
            n = int(v)
            if 0 <= n <= 10_000_000:
                return n
        except Exception:
            pass
    return MIN_PREFIX_WORDS

def resolve_min_prefix_blocks(request: Request) -> int:
    v = request.headers.get("x-min-prefix-blocks") or request.query_params.get("min_prefix_blocks")
    if v:
        try:
            n = int(v)
            if 0 <= n <= 10_000_000:
                return n
        except Exception:
            pass
    return MIN_PREFIX_BLOCKS

def extract_prefix_stats(openai_body: Dict, words_per_block: int) -> Tuple[str, str, List[str], int, int]:
    messages = openai_body.get("messages") or []
    prefix_text = canonical_chat_prefix(messages, add_bos=True)
    key = prefix_key_sha256(prefix_text)
    blocks = block_hashes_from_text(prefix_text, words_per_block)
    words_cnt = len(words_from_text(prefix_text))
    return key, prefix_text, blocks, len(prefix_text), words_cnt

@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {"object": "list", "data": [{"id": DEFAULT_MODEL_ID, "object": "model", "created": now, "owned_by": "local"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream_req = bool(body.get("stream", False))
    model = body.get("model", DEFAULT_MODEL_ID)
    words_per_block = resolve_block_size(request)
    mode = resolve_threshold_mode(request)
    min_chars = resolve_min_prefix_chars(request)
    min_words = resolve_min_prefix_words(request)
    min_blocks = resolve_min_prefix_blocks(request)

    req_key, req_prefix_text, req_blocks, prefix_len, words_cnt = extract_prefix_stats(body, words_per_block)
    blocks_cnt = len(req_blocks)
    log.info(f"request_received model={model} key={req_key} stream={stream_req} prefix_chars={prefix_len} words={words_cnt} blocks={blocks_cnt} wpb={words_per_block} threshold_mode={mode} min_chars={min_chars} min_words={min_words} min_blocks={min_blocks}")

    slot_manager: SlotManager = request.app.state.slot_manager

    small = (
        (mode == "chars" and prefix_len < min_chars) or
        (mode == "words" and words_cnt < min_words) or
        (mode == "blocks" and blocks_cnt < min_blocks)
    )

    if small:
        # Малые запросы — берем глобально свободный/холодный слот, не включаем cache_prompt
        prefer_be = slot_manager._prefer_backend(req_key)
        gk, binding = await slot_manager.acquire_free_or_cold_slot(prefer_backend_id=prefer_be)
        beid, lsid = gk
        backend = request.app.state.backends[beid]
        log.info(f"small_request_use_gslot backend={beid} url={backend.url} local_slot={lsid}")

        if stream_req:
            llama_body = dict(body); llama_body["stream"] = True; llama_body["_slot_id"] = lsid
            async def sse_iterator_small() -> AsyncIterator[bytes]:
                try:
                    async for raw in backend.client.chat_completions_stream(llama_body):
                        yield raw
                except httpx.HTTPError as e:
                    log.warning(f"backend_error url={backend.url} err={e}")
                    yield build_done_chunk()
                    raise HTTPException(status_code=502, detail="llama backend error")
                finally:
                    await slot_manager.mark_slot_cold(gk)
                    try: binding.lock.release()
                    except RuntimeError: pass
                    yield build_done_chunk()
            return StreamingResponse(sse_iterator_small(), media_type="text/event-stream")
        else:
            llama_body = dict(body); llama_body["stream"] = False; llama_body["_slot_id"] = lsid
            try:
                resp = await backend.client.chat_completions_json(llama_body)
                await slot_manager.mark_slot_cold(gk)
                return JSONResponse(content=resp)
            except httpx.HTTPError as e:
                log.warning(f"backend_error url={backend.url} err={e}")
                raise HTTPException(status_code=502, detail="llama backend error")
            finally:
                try: binding.lock.release()
                except RuntimeError: pass

    # Большие запросы — глобальный active/restore/cold
    gk, binding, source, lcp_count, binding_total = await slot_manager.ensure_slot_for_request(req_key, req_prefix_text, req_blocks, words_per_block)
    beid, lsid = gk
    backend = request.app.state.backends[beid]
    log.info(f"match_info source={source} backend={beid} url={backend.url} gslot={gk} lcp_blocks={lcp_count}/{blocks_cnt} binding_blocks={binding_total}")

    if stream_req:
        llama_body = dict(body); llama_body["stream"] = True; llama_body["cache_prompt"] = True; llama_body["_slot_id"] = lsid
        async def sse_iterator() -> AsyncIterator[bytes]:
            try:
                async for raw in backend.client.chat_completions_stream(llama_body):
                    yield raw
                    await slot_manager.touch(gk)
            except httpx.HTTPError as e:
                log.warning(f"backend_error url={backend.url} err={e}")
                yield build_done_chunk()
                raise HTTPException(status_code=502, detail="llama backend error")
            finally:
                try: binding.lock.release()
                except RuntimeError: pass
                yield build_done_chunk()
        return StreamingResponse(sse_iterator(), media_type="text/event-stream")
    else:
        llama_body = dict(body); llama_body["stream"] = False; llama_body["cache_prompt"] = True; llama_body["_slot_id"] = lsid
        try:
            resp = await backend.client.chat_completions_json(llama_body)
            return JSONResponse(content=resp)
        except httpx.HTTPError as e:
            log.warning(f"backend_error url={backend.url} err={e}")
            raise HTTPException(status_code=502, detail="llama backend error")
        finally:
            try: binding.lock.release()
            except RuntimeError: pass
