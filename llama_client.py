# llama_client.py

# -*- coding: utf-8 -*-

"""
HTTP-клиент к llama.cpp: /v1/chat/completions (stream/non-stream), /slots save/restore, /v1/models.

- stream: build_request+send(stream=True), сырые байты.
- non-stream: строгий JSON парсинг + fallback, если content-type не JSON.
- /slots: filename в JSON-теле (во избежание 500 parse error).
- Пин слота дублируется в root/options/query.
- get_model_id(): получает текущий id модели с /v1/models.
"""

import httpx
import logging
from typing import Dict, Optional, Tuple

from config import REQUEST_TIMEOUT

log = logging.getLogger(__name__)


class LlamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=REQUEST_TIMEOUT,
            limits=limits,
        )
        log.info("client_init url=%s httpx_version=%s", base_url, httpx.__version__)

    async def close(self):
        await self.client.aclose()

    @staticmethod
    def _with_slot_id(body: Dict, slot_id: Optional[int]) -> Tuple[Dict, Dict]:
        if slot_id is None:
            return body, {}

        new_body = dict(body)

        # root
        new_body["_slot_id"] = slot_id
        new_body["slot_id"] = slot_id
        new_body["id_slot"] = slot_id

        # options
        opts = dict(new_body.get("options") or {})
        opts["slot_id"] = slot_id
        opts["id_slot"] = slot_id
        new_body["options"] = opts

        # query
        query = {"slot_id": slot_id, "id_slot": slot_id}
        return new_body, query

    async def chat_completions(
        self,
        body: Dict,
        slot_id: Optional[int] = None,
        stream: bool = False,
    ):
        body2, query = self._with_slot_id(body, slot_id)

        if stream:
            req = self.client.build_request(
                "POST",
                "/v1/chat/completions",
                json=body2,
                params=query,
            )
            resp = await self.client.send(req, stream=True)
            return resp

        resp = await self.client.post(
            "/v1/chat/completions",
            json=body2,
            params=query,
        )
        resp.raise_for_status()

        ctype = resp.headers.get("content-type", "")
        if "application/json" not in ctype:
            raw = resp.text or ""
            log.error(
                "non_stream_non_json content_type=%s raw_len=%d",
                ctype,
                len(raw),
            )
            return {
                "object": "error",
                "message": "provider returned non-JSON",
                "raw": raw[:2048],
            }

        try:
            return resp.json()
        except Exception as e:
            raw = resp.text or ""
            log.error(
                "non_stream_json_parse_error status=%d raw_len=%d err=%s",
                resp.status_code,
                len(raw),
                e,
            )
            return {
                "object": "error",
                "message": "invalid json from provider",
                "raw": raw[:2048],
            }

    async def save_slot(self, slot_id: int, basename: str) -> bool:
        # JSON body: {"filename": "..."} — иначе 500 на некоторых сборках
        resp = await self.client.post(
            f"/slots/{slot_id}",
            params={"action": "save"},
            json={"filename": basename},
        )

        if resp.status_code == 500:
            log.warning(
                "save_slot_500 slot=%d basename=%s",
                slot_id,
                basename[:16],
            )
            return False

        resp.raise_for_status()
        return True

    async def restore_slot(self, slot_id: int, basename: str) -> bool:
        resp = await self.client.post(
            f"/slots/{slot_id}",
            params={"action": "restore"},
            json={"filename": basename},
        )

        if resp.status_code != 200:
            log.warning(
                "restore_slot_status=%d slot=%d basename=%s",
                resp.status_code,
                slot_id,
                basename[:16],
            )
            return False

        return True

    async def get_model_id(self) -> str:
        """
        Получает id модели у конкретного llama.cpp через /v1/models.

        Используется только для внутреннего кеширования (ключи файлов/мета),
        наружу прокси продолжает отдавать MODEL_ID из своей конфигурации.
        """
        try:
            resp = await self.client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data") or []
            if models and isinstance(models[0], dict):
                mid = models[0].get("id") or "unknown"
            else:
                mid = "unknown"
            log.debug("get_model_id base_url=%s id=%s", self.base_url, mid)
            return mid
        except Exception as e:
            log.warning("get_model_id_fail base_url=%s err=%s", self.base_url, e)
            return "unknown"
