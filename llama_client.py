# llama_client.py
# -*- coding: utf-8 -*-

"""
HTTP-клиент для взаимодействия с llama.cpp server.

Назначение:
- Обращается к /v1/chat/completions (JSON и SSE stream с префлайтом).
- Управляет KV-кэшем через /slots и /slots/{id}?action=save|restore (basename -> серверный путь по --slot-save-path).
- Дублирует _slot_id в теле, options и query-параметрах, чтобы повысить шанс закрепить запрос за нужным слотом.

Инварианты:
- На каждый backend создаётся свой LlamaClient (base_url уникален).
- Таймауты и лимиты соединений задаются через config.REQUEST_TIMEOUT и httpx.Limits.
"""

from typing import Dict, Optional, Tuple
import logging
import httpx

from config import REQUEST_TIMEOUT

log = logging.getLogger(__name__)

class LlamaClient:
    """
    Инкапсуляция HTTP-запросов к конкретному экземпляру llama.cpp.
    """

    def __init__(self, base_url: str):
        """
        Создаёт асинхронный httpx клиент с заданным base_url и преднастроенными лимитами.

        Аргументы:
        - base_url: адрес llama.cpp server (например, http://127.0.0.1:8000).
        """
        self._base_url = base_url
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            limits=limits,
        )
        log.info("client_init base_url=%s", base_url)

    async def aclose(self):
        """
        Аккуратно закрыть все соединения клиента (вызывается при остановке приложения).
        """
        await self._client.aclose()
        log.info("client_closed base_url=%s", self._base_url)

    @staticmethod
    def _with_slot_id(body: Dict, slot_id: Optional[int]) -> Tuple[Dict, Dict]:
        """
        Если передан slot_id, дублирует его:
        - в корень тела (_slot_id, slot_id, id_slot),
        - в options.slot_id/id_slot,
        - в query (?slot_id=&id_slot=).

        Возвращает:
        - (изменённое тело, query-параметры).
        """
        if slot_id is None:
            return body, {}
        new_body = dict(body)
        new_body["_slot_id"] = slot_id
        new_body["slot_id"] = slot_id
        new_body["id_slot"] = slot_id
        opts = dict(new_body.get("options") or {})
        opts["slot_id"] = slot_id
        opts["id_slot"] = slot_id
        new_body["options"] = opts
        query = {"slot_id": slot_id, "id_slot": slot_id}
        return new_body, query

    async def get_slots(self) -> Dict:
        """
        Получить статус слотов от сервера (диагностика/отладка).

        Возвращает:
        - JSON статус слотов.
        """
        r = await self._client.get("/slots")
        r.raise_for_status()
        data = r.json()
        log.debug("slots_status status=%d", r.status_code)
        return data

    async def save_slot(self, slot_id: int, basename: str) -> Dict:
        """
        Сохранить KV-кэш выбранного слота на диск (сервер llama.cpp).

        Аргументы:
        - slot_id: локальный номер слота на сервере.
        - basename: имя файла без пути (сервер добавляет расширение и путь сам).

        Возвращает:
        - JSON результата (если есть).
        """
        log.info("slot_save slot_id=%d basename=%s", slot_id, basename[:16])
        r = await self._client.post(f"/slots/{slot_id}", params={"action": "save", "filename": basename})
        r.raise_for_status()
        data = r.json()
        log.info("slot_saved slot_id=%d basename=%s status=%d", slot_id, basename[:16], r.status_code)
        return data

    async def restore_slot(self, slot_id: int, basename: str) -> Dict:
        """
        Восстановить KV-кэш выбранного слота с диска по basename (server-side --slot-save-path).

        Аргументы:
        - slot_id: локальный номер слота.
        - basename: имя файла без пути.

        Возвращает:
        - JSON результата (если есть).
        """
        log.info("slot_restore slot_id=%d basename=%s", slot_id, basename[:16])
        r = await self._client.post(f"/slots/{slot_id}", params={"action": "restore", "filename": basename})
        r.raise_for_status()
        data = r.json()
        log.info("slot_restored slot_id=%d basename=%s status=%d", slot_id, basename[:16], r.status_code)
        return data

    async def chat_completions_json(self, body: Dict, slot_id: Optional[int] = None) -> Dict:
        """
        Выполнить JSON-вызов /v1/chat/completions (stream=false).

        Аргументы:
        - body: тело OpenAI-совместимого запроса.
        - slot_id: локальный слот (опционально).

        Возвращает:
        - JSON-ответ от сервера.
        """
        body2, query = self._with_slot_id(body, slot_id)
        log.info("chat_json begin slot_id=%s stream=%s", slot_id, bool(body2.get("stream")))
        r = await self._client.post("/v1/chat/completions", json=body2, params=query)
        log.info("chat_json end status=%d slot_id=%s", r.status_code, slot_id)
        r.raise_for_status()
        return r.json()

    async def open_chat_stream(self, body: Dict, slot_id: Optional[int] = None) -> httpx.Response:
        """
        Префлайт для stream: вернуть открытый Response (stream=True), чтобы проверить статус до старта SSE.

        Аргументы:
        - body: тело запроса.
        - slot_id: локальный слот.

        Возвращает:
        - httpx.Response с открытым стримом; вызывающий обязан закрыть r.aclose().
        """
        body2, query = self._with_slot_id(body, slot_id)
        log.info("chat_stream open slot_id=%s", slot_id)
        req = self._client.build_request("POST", "/v1/chat/completions", json=body2, params=query)
        r = await self._client.send(req, stream=True)
        log.info("chat_stream preflight status=%d slot_id=%s", r.status_code, slot_id)
        return r
