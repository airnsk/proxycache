# config.py
# -*- coding: utf-8 -*-
import os
import json
import logging

# По умолчанию — один бэкенд (совместимость с прошлой версией)
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8000")
SLOTS_COUNT = int(os.getenv("SLOTS_COUNT", "4"))

# Мульти-бэкенды: JSON в LLAMA_BACKENDS, например:
# [{"url":"http://127.0.0.1:8000","slots":4},{"url":"http://127.0.0.1:8001","slots":8}]
LLAMA_BACKENDS_JSON = os.getenv("LLAMA_BACKENDS", "").strip()
if LLAMA_BACKENDS_JSON:
    try:
        BACKENDS = json.loads(LLAMA_BACKENDS_JSON)
    except Exception:
        BACKENDS = []
else:
    BACKENDS = [{"url": LLAMA_SERVER_URL, "slots": SLOTS_COUNT}]

# Таймаут HTTP‑клиента
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

# Системный промпт (опционально, для канонизации префикса)
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE")

# Идентификатор модели
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "llama.cpp")

# Прибитые ключи, недоступные для эвикции
PINNED_PREFIX_KEYS = set(json.loads(os.getenv("PINNED_KEYS", "[]")))

# Размер блока по словам для LCP
DEFAULT_WORDS_PER_BLOCK = int(os.getenv("WORDS_PER_BLOCK", "16"))

# Лимит сканирования .meta
DISK_META_SCAN_LIMIT = int(os.getenv("DISK_META_SCAN_LIMIT", "200"))

# Порог «малых» запросов
THRESHOLD_MODE = os.getenv("THRESHOLD_MODE", "chars").lower()
MIN_PREFIX_CHARS = int(os.getenv("MIN_PREFIX_CHARS", "5000"))
MIN_PREFIX_WORDS = int(os.getenv("MIN_PREFIX_WORDS", "1000"))
MIN_PREFIX_BLOCKS = int(os.getenv("MIN_PREFIX_BLOCKS", "20"))

# Единый порог схожести для active‑lcp и restore‑lcp
SIMILARITY_MIN_RATIO = float(os.getenv("SIMILARITY_MIN_RATIO", "0.85"))

# Каталог локальных .meta (индекса прокси)
LOCAL_META_DIR = os.getenv("LOCAL_META_DIR", "./kvslots_meta")

# Опционально: локальный путь, смонтированный к --slot-save-path (для логирования размеров .bin)
SLOT_SAVE_MOUNT = os.getenv("SLOT_SAVE_MOUNT")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
