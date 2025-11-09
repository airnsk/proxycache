# proxycache.py
# -*- coding: utf-8 -*-

"""
Точка запуска сервиса через uvicorn.

Назначение:
- Импортирует FastAPI-приложение из app.py и запускает HTTP-сервер.
- Параметры HOST/PORT читаются из окружения и логируются при старте.
"""

import os
import logging
import uvicorn

from app import app  # FastAPI instance

log = logging.getLogger(__name__)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8081"))
    log.info("server_start host=%s port=%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
