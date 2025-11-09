# llama_client.py
# -*- coding: utf-8 -*-
import httpx
from typing import AsyncIterator, Dict, Any
from config import REQUEST_TIMEOUT

class LlamaClient:
    """
    Тонкий HTTP‑клиент к llama.cpp HTTP серверу (один экземпляр).
    Предоставляет chat completions (SSE/JSON) и слотовые операции save/restore (basename).
    """
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    async def close(self):
        await self._client.aclose()

    # Потоковый чат (SSE сквозной проксирование)
    async def chat_completions_stream(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        url = f"{self.base_url}/v1/chat/completions"
        async with self._client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_raw():
                yield chunk

    # Нестрёминговый чат (JSON)
    async def chat_completions_json(self, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        r = await self._client.post(url, json=body)
        r.raise_for_status()
        return r.json()

    # Сохранение слота (basename файла; сервер пишет в --slot-save-path)
    async def save_slot(self, slot_id: int, filename_basename: str) -> Dict[str, Any]:
        url = f"{self.base_url}/slots/{slot_id}"
        r = await self._client.post(url, params={"action": "save"}, json={"filename": filename_basename})
        r.raise_for_status()
        return r.json()

    # Восстановление слота (basename файла)
    async def restore_slot(self, slot_id: int, filename_basename: str) -> Dict[str, Any]:
        url = f"{self.base_url}/slots/{slot_id}"
        r = await self._client.post(url, params={"action": "restore"}, json={"filename": filename_basename})
        r.raise_for_status()
        return r.json()
