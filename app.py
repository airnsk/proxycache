# app.py
# -*- coding: utf-8 -*-

"""
FastAPI-приложение: OpenAI-совместимый прокси поверх кластера llama.cpp.

Назначение:
- Предоставляет /v1/models и /v1/chat/completions.
- Управляет раздачей запросов на несколько бэкендов llama.cpp с глобальным пулом слотов.
- Отрабатывает «малые» и «большие» запросы по общей семантике (пороговые эвристики и LCP/restore),
  при этом не ломая поведение single-backend варианта.

Как работает:
- В lifespan поднимаются клиенты LlamaClient на каждый backend и инициализируется SlotManager с глобальным пулом.
- Запрос /v1/chat/completions:
  1) Канонизирует историю сообщений, строит key/blocks, определяет «малость» по THRESHOLD_MODE.
  2) Для «малых»: выбирает свободный/холодный слот (с приоритетом предпочитаемого бэкенда), не включает cache_prompt,
     по окончании помечает слот cold и освобождает lock.
  3) Для «больших»: вызывает ensure_slot_for_request (active-exact → active-lcp → restore-lcp → cold),
     включает cache_prompt, в stream режиме периодически touch для LRU, в finally освобождает lock.
- Для stream используется «префлайт»: сначала открывается SSE при помощи LlamaClient.open_chat_stream,
  статус проверяется до отправки заголовков StreamingResponse, чтобы корректно отработать ошибки.

Исправление нестримового пути:
- Всегда подставляется DEFAULT_MODEL_ID, если клиент не прислал model, чтобы исключить 4xx/зависания на стороне llama.cpp.
- В JSON-пути добавлена явная обработка httpx.HTTPError с возвратом JSON ошибки и логированием,
  чтобы не допускать «молчаливых» зависаний при ошибках сети/сервера.
- Логи печатают ключевые метрики, выбор backend/slot, источник назначения и длительность обработки.

Логирование:
- Для каждого запроса печатаются key (сокращённый), stream, wpb, длины chars/words/blocks, классификация small/large,
  выбранный backend/slot и их состояние, source назначения, статусы префлайта и длительность обработки.
"""

import asyncio
import time
import logging
import httpx  # для перехвата сетевых/HTTP ошибок в JSON-пути
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from config import (
    LLAMA_BACKENDS,
    DEFAULT_MODEL_ID,
    DEFAULT_WORDS_PER_BLOCK,
    THRESHOLD_MODE,
    MIN_PREFIX_CHARS,
    MIN_PREFIX_WORDS,
    MIN_PREFIX_BLOCKS,
    SYSTEM_PROMPT_FILE,
    ADD_BOS,
)
import hashing as hs
from llama_client import LlamaClient
from slot_manager import SlotManager, Backend

log = logging.getLogger(__name__)
app = FastAPI(title="llama.cpp proxy with multi-backend")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Инициализация ресурсов приложения:
    - Создание LlamaClient по каждому backend.url из конфигурации.
    - Формирование списка Backend и инициализация SlotManager.
    - Сохранение ссылок в app.state и корректное закрытие клиентов на остановке.
    """
    clients: List[LlamaClient] = []
    backends: List[Backend] = []
    for idx, be in enumerate(LLAMA_BACKENDS):
        c = LlamaClient(base_url=be["url"])
        clients.append(c)
        backends.append(Backend(id=idx, url=be["url"], slots=int(be["slots"]), client=c))

    sm = SlotManager(backends=backends, model_id=DEFAULT_MODEL_ID)

    app.state.clients = clients
    app.state.backends = backends
    app.state.slot_manager = sm

    log.info("app_lifespan_start backends=%d", len(backends))
    try:
        yield
    finally:
        await asyncio.gather(*[c.aclose() for c in clients])
        log.info("app_lifespan_stop")

app.router.lifespan_context = lifespan

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-совместимый список моделей (минимально достаточный для клиентов).
    """
    return {"object": "list", "data": [{"id": DEFAULT_MODEL_ID, "object": "model"}]}

def _is_small_request(prefix_text: str, blocks: List[str]) -> bool:
    """
    Определить «малость» запроса согласно THRESHOLD_MODE:
    - chars: по длине текста в символах,
    - blocks: по количеству блоков,
    - words (по умолчанию): по количеству слов.
    """
    if THRESHOLD_MODE == "chars":
        return len(prefix_text) < MIN_PREFIX_CHARS
    elif THRESHOLD_MODE == "blocks":
        return len(blocks) < MIN_PREFIX_BLOCKS
    else:
        return len(hs.words_from_text(prefix_text)) < MIN_PREFIX_WORDS

def _build_body(base: Dict[str, Any], stream: bool, cache_prompt: bool, slot_id: int | None) -> Dict[str, Any]:
    """
    Собрать тело запроса к llama.cpp:
    - Проставить stream,
    - Включить options.cache_prompt для «больших»,
    - Проставить локальный slot_id/id_slot в options (дублирование также делается в клиенте).
    """
    body = dict(base)
    body["stream"] = bool(stream)
    # Гарантируем, что model присутствует (фикс для нестримовых зависаний)
    if not body.get("model"):
        body["model"] = DEFAULT_MODEL_ID
    opts = dict(body.get("options") or {})
    if cache_prompt:
        opts["cache_prompt"] = True
    if slot_id is not None:
        opts["slot_id"] = slot_id
        opts["id_slot"] = slot_id
    body["options"] = opts
    return body

async def _stream_sse(resp) -> AsyncIterator[bytes]:
    """
    Итерировать сырые чанки из httpx.Response (stream=True) для StreamingResponse.
    """
    async for chunk in resp.aiter_raw():
        if chunk:
            yield chunk

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """
    Основная точка входа OpenAI-совместимого API:
    - Канонизация/классификация,
    - Выбор слота/бэкенда,
    - Прокси JSON или SSE (с префлайтом),
    - Корректное освобождение lock/пометка cold/обновление LRU.
    """
    t0 = time.time()
    state = app.state
    sm: SlotManager = state.slot_manager

    data = await req.json()
    # Подстраховка: если клиент не прислал model — подставим DEFAULT_MODEL_ID (важно для нестримовых путей)
    if not data.get("model"):
        data["model"] = DEFAULT_MODEL_ID

    messages = data.get("messages") or []
    model = data.get("model") or DEFAULT_MODEL_ID
    wpb = int(data.get("words_per_block") or DEFAULT_WORDS_PER_BLOCK)
    stream = bool(data.get("stream") or False)

    # Канонизация и статистика
    prefix_text = hs.canonical_chat_prefix(messages, SYSTEM_PROMPT_FILE, ADD_BOS)
    req_blocks = hs.block_hashes_from_text(prefix_text, wpb)
    key = hs.prefix_key_sha256(prefix_text)
    words = hs.words_from_text(prefix_text)

    chars_len = len(prefix_text)
    words_len = len(words)
    blocks_len = len(req_blocks)
    is_small = _is_small_request(prefix_text, req_blocks)

    log.info(
        "request key=%s model=%s stream=%s chars=%d words=%d blocks=%d wpb=%d small=%s mode=%s",
        key[:16], model, stream, chars_len, words_len, blocks_len, wpb, is_small, THRESHOLD_MODE,
    )

    if is_small:
        # Малые — без cache_prompt; слот помечается как cold по завершении
        prefer_be = sm._prefer_backend(key)
        gslot, lock = await sm.acquire_free_or_cold_slot(prefer_backend_id=prefer_be)
        await lock.acquire()
        backend_id, local_slot_id = gslot
        try:
            be = state.backends[backend_id]
            client = be.client
            body = _build_body(data, stream=stream, cache_prompt=False, slot_id=local_slot_id)
            log.info("dispatch_small be=%d url=%s slot=%d state=free_or_cold", backend_id, be.url, local_slot_id)

            if stream:
                # Префлайт перед SSE
                resp = await client.open_chat_stream(body, slot_id=local_slot_id)
                if resp.status_code != 200:
                    try:
                        err = await resp.aread()
                    finally:
                        await resp.aclose()
                    log.error("stream_preflight_error be=%d slot=%d status=%d", backend_id, local_slot_id, resp.status_code)
                    return JSONResponse(status_code=resp.status_code, content={"error": err.decode("utf-8", "ignore")})
                async def gen():
                    log.info("stream_begin be=%d slot=%d", backend_id, local_slot_id)
                    try:
                        async for chunk in _stream_sse(resp):
                            yield chunk
                    finally:
                        await resp.aclose()
                        log.info("stream_end be=%d slot=%d", backend_id, local_slot_id)
                return StreamingResponse(gen(), media_type="text/event-stream")
            else:
                try:
                    out = await client.chat_completions_json(body, slot_id=local_slot_id)
                    log.info("json_complete be=%d slot=%d", backend_id, local_slot_id)
                    return JSONResponse(content=out)
                except httpx.HTTPError as e:
                    # Явный ответ об ошибке в JSON-режиме, чтобы не казалось «не отвечает»
                    log.exception("json_error be=%d slot=%d err=%s", backend_id, local_slot_id, str(e))
                    return JSONResponse(status_code=502, content={"error": str(e)})
        finally:
            sm.mark_slot_cold(gslot)
            sm.release(gslot)
            log.info("finalize_small be=%d slot=%d duration_ms=%d", backend_id, local_slot_id, int((time.time() - t0) * 1000))

    # Большие — cache_prompt=True, ensure_slot_for_request
    gslot = None
    backend_id = None
    local_slot_id = None
    try:
        gslot, binding, source, lcp_cnt, total = await sm.ensure_slot_for_request(
            req_key=key,
            prefix_text=prefix_text,
            req_blocks=req_blocks,
            words_per_block=wpb,
        )
        backend_id, local_slot_id = gslot
        be = state.backends[backend_id]
        client = be.client

        log.info("dispatch_big source=%s be=%d url=%s slot=%d lcp=%d bindings=%d", source, backend_id, be.url, local_slot_id, lcp_cnt, total)

        body = _build_body(data, stream=stream, cache_prompt=True, slot_id=local_slot_id)

        if stream:
            resp = await client.open_chat_stream(body, slot_id=local_slot_id)
            if resp.status_code != 200:
                try:
                    err = await resp.aread()
                finally:
                    await resp.aclose()
                log.error("stream_preflight_error be=%d slot=%d status=%d", backend_id, local_slot_id, resp.status_code)
                return JSONResponse(status_code=resp.status_code, content={"error": err.decode("utf-8", "ignore")})
            async def gen():
                log.info("stream_begin be=%d slot=%d", backend_id, local_slot_id)
                try:
                    async for chunk in _stream_sse(resp):
                        sm.touch(gslot)  # поддерживаем LRU
                        yield chunk
                finally:
                    await resp.aclose()
                    log.info("stream_end be=%d slot=%d", backend_id, local_slot_id)
            return StreamingResponse(gen(), media_type="text/event-stream")
        else:
            try:
                out = await client.chat_completions_json(body, slot_id=local_slot_id)
                log.info("json_complete be=%d slot=%d", backend_id, local_slot_id)
                return JSONResponse(content=out)
            except httpx.HTTPError as e:
                # Возврат JSON-ошибки для прозрачности клиента
                log.exception("json_error be=%d slot=%d err=%s", backend_id, local_slot_id, str(e))
                return JSONResponse(status_code=502, content={"error": str(e)})
    finally:
        if gslot is not None:
            sm.release(gslot)
            log.info("finalize_big be=%d slot=%d duration_ms=%d", backend_id, local_slot_id, int((time.time() - t0) * 1000))
