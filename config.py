# config.py
# -*- coding: utf-8 -*-

"""
Конфигурация прокси-сервиса (OpenAI-совместимый прокси поверх нескольких экземпляров llama.cpp).

Назначение:
- Единый слой конфигурации для single-backend и multi-backend режимов.
- Управление порогами (малые запросы: chars/words/blocks), порогом похожести LCP (SIMILARITY_MIN_RATIO).
- Пути для локальных .meta индексов прокси и базовые настройки логирования.

Как работает:
- LLAMA_BACKENDS включает multi-backend (JSON: [{ "url": "...", "slots": N }]); если не задан — fallback на
  LLAMA_SERVER_URL + SLOTS_COUNT (режим одного сервера), благодаря чему архитектура не ломается.
- Параметры MIN_PREFIX_* и THRESHOLD_MODE определяют «малость» запроса: малые не кэшируют префикс и помечают слот cold.
- Для кросс-бэкенд restore обязательно одинаковый --slot-save-path на всех llama.cpp, т.к. клиент оперирует basename.

Логирование:
- LOG_LEVEL/LOG_FORMAT/LOG_DATEFMT управляют форматированием; key=value стиль поддерживается форматами логов в коде.

Инварианты:
- SIMILARITY_MIN_RATIO применяется одинаково к active-lcp и restore-lcp, exact совпадение всегда принимается.
- PINNED_PREFIX_KEYS запрещает эвикт «важных» ключей префикса.
"""

import os
import json
import logging

# Базовый адрес llama.cpp server (fallback при одном бекенде)
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8000")

# Число слотов на сервере (fallback при одном бекенде)
SLOTS_COUNT = int(os.getenv("SLOTS_COUNT", "4"))

# Мульти-бекенды: список словарей {url, slots}
_LLAMA_BACKENDS_RAW = os.getenv("LLAMA_BACKENDS")
LLAMA_BACKENDS = None
if _LLAMA_BACKENDS_RAW:
    try:
        LLAMA_BACKENDS = json.loads(_LLAMA_BACKENDS_RAW)
        assert isinstance(LLAMA_BACKENDS, list)
        for be in LLAMA_BACKENDS:
            assert isinstance(be, dict) and "url" in be and "slots" in be
            be["slots"] = int(be["slots"])
    except Exception as e:
        logging.warning("Failed to parse LLAMA_BACKENDS, falling back to single backend: %s", e)
        LLAMA_BACKENDS = None

if LLAMA_BACKENDS is None:
    LLAMA_BACKENDS = [{"url": LLAMA_SERVER_URL, "slots": SLOTS_COUNT}]

# Таймауты HTTP‑клиента (сек)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

# Системный промпт (опционально, для канонизации префикса)
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE")

# Идентификатор модели для /v1/models
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "llama.cpp")

# Прибитые ключи, недоступные для эвикции
try:
    PINNED_PREFIX_KEYS = set(json.loads(os.getenv("PINNED_KEYS", "[]")))
except Exception:
    PINNED_PREFIX_KEYS = set()

# Размер блока по словам для блочной эвристики (обычно 16)
DEFAULT_WORDS_PER_BLOCK = int(os.getenv("WORDS_PER_BLOCK", "16"))

# Лимит сканирования .meta на диске
DISK_META_SCAN_LIMIT = int(os.getenv("DISK_META_SCAN_LIMIT", "200"))

# Каталог локальных индексов (.meta) у прокси
LOCAL_META_DIR = os.getenv("LOCAL_META_DIR", "./.proxy_meta")

# Необязательный локальный просмотр каталога server‑side кэшей (для отладки)
SLOT_SAVE_MOUNT = os.getenv("SLOT_SAVE_MOUNT")  # например, монтирование NFS

# Порог похожести LCP (0..1) для active‑lcp и restore‑lcp
SIMILARITY_MIN_RATIO = float(os.getenv("SIMILARITY_MIN_RATIO", "0.6"))

# Порог режимов «малых» запросов: chars|words|blocks
THRESHOLD_MODE = os.getenv("THRESHOLD_MODE", "words")  # "chars" | "words" | "blocks"
MIN_PREFIX_CHARS = int(os.getenv("MIN_PREFIX_CHARS", "400"))
MIN_PREFIX_WORDS = int(os.getenv("MIN_PREFIX_WORDS", "100"))
MIN_PREFIX_BLOCKS = int(os.getenv("MIN_PREFIX_BLOCKS", "8"))

# Управление BOS и канонизацией (опционально)
ADD_BOS = os.getenv("ADD_BOS", "false").lower() in ("1", "true", "yes")

# Логирование: единый формат для всех модулей
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s - %(message)s")
LOG_DATEFMT = os.getenv("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), format=LOG_FORMAT, datefmt=LOG_DATEFMT)
