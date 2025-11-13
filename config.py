# config.py
# -*- coding: utf-8 -*-

"""
Единая конфигурация для simple_proxycache:
- BACKENDS: [{"url": "...", "n_slots": N}]
- WORDS_PER_BLOCK, BIG_THRESHOLD_WORDS, LCP_TH
- PORT, REQUEST_TIMEOUT, MODEL_ID
"""

import os
import json
import logging

# Backends
BACKENDS_RAW = os.getenv("BACKENDS")
if BACKENDS_RAW:
    try:
        BACKENDS = json.loads(BACKENDS_RAW)
    except Exception:
        BACKENDS = []
else:
    BACKENDS = [{
        "url": os.getenv("LLAMA_URL", "http://127.0.0.1:8000"),
        "n_slots": int(os.getenv("N_SLOTS", "2")),
    }]

# Words per block for LCP
WORDS_PER_BLOCK = int(os.getenv("WORDS_PER_BLOCK", "100"))

# Big request threshold
BIG_THRESHOLD_WORDS = int(os.getenv("BIG_THRESHOLD_WORDS", "500"))

# LCP threshold (0..1)
LCP_TH = float(os.getenv("LCP_TH", "0.6"))

# Meta dir
META_DIR = os.getenv("META_DIR", "./kv_meta")
os.makedirs(META_DIR, exist_ok=True)

# HTTP timeout
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

# Model id
MODEL_ID = os.getenv("MODEL_ID", "llama.cpp")

# Service port
PORT = int(os.getenv("PORT", "8081"))

# Logs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Model id (for compatibility)
DEFAULT_MODEL_ID = MODEL_ID

# Meta scanning limits
DISK_META_SCAN_LIMIT = int(os.getenv("DISK_META_SCAN_LIMIT", "1000"))

# Pinned keys configuration
PINNED_PREFIX_KEYS = os.getenv("PINNED_PREFIX_KEYS", "[]")
try:
    PINNED_PREFIX_KEYS = json.loads(PINNED_PREFIX_KEYS)
except Exception:
    PINNED_PREFIX_KEYS = []

# Slot cache directory
SLOT_SAVE_MOUNT = os.getenv("SLOT_SAVE_MOUNT", "./slotcache")

# Similarity threshold for LCP matching
SIMILARITY_MIN_RATIO = float(os.getenv("SIMILARITY_MIN_RATIO", "0.6"))
