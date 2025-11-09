# hashing.py
# -*- coding: utf-8 -*-

"""
Хэширование и канонизация префиксов.

Назначение:
- Превращает OpenAI-совместимые messages в детерминированный канонический текст.
- Выполняет блочную эвристику: Формирует цепочку SHA-256 для блоков по словам.
- Быстро оценивает похожесть префиксов через LCP (длину общего префикса цепочек блоков).
- Ведёт локальный индекс .meta у прокси, чтобы быстро подбирать restore-кандидатов с диска.

Ключевые функции:
- normalize_content: безопасно извлекает текст из content (строка или parts).
- canonical_chat_prefix: строит стабильную канонизацию чата с ролями и системным промптом.
- block_hashes_from_text: разбивает текст на блоки (words_per_block) и считает SHA-256 каждого блока.
- lcp_blocks: возвращает длину общего префикса по двум массивам блок-хэшей.
- prefix_key_sha256: формирует ключ префикса (basename) — хэш канонического текста.
- write_meta_for_key_local / scan_all_meta_local: запись и сканирование локальной .meta.

Инварианты:
- Формат канонизации минималистичен и стабилен для одинакового ввода.
- .meta хранится только у прокси (LOCAL_META_DIR), а KV-файлы — у llama.cpp (по --slot-save-path).
"""

import os
import re
import hashlib
import time
import glob
import json
import logging
from typing import List, Dict, Optional

from config import SYSTEM_PROMPT_FILE, LOCAL_META_DIR

log = logging.getLogger(__name__)

# Гарантируем существование каталога для .meta (локальные метаданные прокси)
os.makedirs(LOCAL_META_DIR, exist_ok=True)

def normalize_content(content) -> str:
    """
    Нормализация OpenAI content.

    Аргументы:
    - content: строка или список частей [{"type":"text","text":"..."}].

    Возвращает:
    - Строку с объединённым текстом; для неизвестных типов — безопасная строковая форма.

    Особенности:
    - Игнорирует не-текстовые части (image_url и т.п.).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        s = content.strip()
        return s
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text")
                if isinstance(t, str):
                    t = t.strip()
                if t:
                    parts.append(t)
        s = " ".join(parts).strip()
        return s
    try:
        s = str(content).strip()
        return s
    except Exception:
        return ""

def _read_system_prompt(system_prompt_file: Optional[str]) -> str:
    """
    Безопасно прочитать системный промпт с диска.

    Возвращает:
    - Текст системного промпта или пустую строку при ошибке/отсутствии файла.
    """
    if not system_prompt_file:
        return ""
    try:
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return txt
    except Exception:
        return ""

def canonical_chat_prefix(messages: List[Dict], system_prompt_file: Optional[str], add_bos: bool) -> str:
    """
    Канонизация истории чата в единый текст с учётом роли и системного промпта.

    Аргументы:
    - messages: список OpenAI-сообщений {role, content}.
    - system_prompt_file: путь к системному промпту (опционально).
    - add_bos: добавить <BOS> в начало (для некоторых сборок это полезно).

    Возвращает:
    - Канонический текст префикса.

    Формат:
    - [system]: <text>
    - [user]: <text>
    - [assistant]: <text>
    """
    parts = []
    sys_prompt = _read_system_prompt(system_prompt_file)
    if sys_prompt:
        parts.append(f"[system]: {sys_prompt}")

    for m in messages or []:
        role = (m.get("role") or "").strip().lower()
        content = normalize_content(m.get("content"))
        if not content:
            continue
        if role not in ("system", "user", "assistant"):
            role = "user"
        parts.append(f"[{role}]: {content}")

    text = "\n".join(parts).strip()
    if add_bos:
        text = "<BOS>\n" + text
    log.debug("canon_prefix chars=%d", len(text))
    return text

_WORD_RE = re.compile(r"[^\s]+")

def words_from_text(text: str) -> List[str]:
    """
    Разбиение текста на «слова» по пробелам (быстро для порогов и блочного разбиения).
    """
    ws = _WORD_RE.findall(text or "")
    return ws

def block_hashes_from_text(text: str, words_per_block: int) -> List[str]:
    """
    Разбивает текст на блоки по words_per_block и возвращает SHA-256 каждого блока.

    Возвращает:
    - Список hex-хэшей блоков (цепочка, пригодная для LCP сравнения без токенизации).
    """
    ws = words_from_text(text)
    blocks = []
    for i in range(0, len(ws), max(1, words_per_block)):
        block = " ".join(ws[i:i+words_per_block])
        digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
        blocks.append(digest)
    log.debug("blocks words=%d blocks=%d wpb=%d", len(ws), len(blocks), words_per_block)
    return blocks

def lcp_blocks(a: List[str], b: List[str]) -> int:
    """
    Длина общего префикса по двум массивам блок-хэшей (в блоках).
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def prefix_key_sha256(prefix_text: str) -> str:
    """
    Ключ префикса — SHA-256 от канонического текста; используется как basename и имя .meta.
    """
    key = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()
    log.debug("prefix_key key=%s", key[:16])
    return key

def _meta_path_for_key(key: str) -> str:
    """
    Построить путь к локальному .meta файлу по key (у прокси, не на сервере).
    """
    return os.path.join(LOCAL_META_DIR, f"{key}.meta.json")

def write_meta_for_key_local(
    key: str,
    prefix_text: str,
    model_id: str,
    words_per_block: int,
    block_hashes: List[str],
) -> None:
    """
    Записать локальное .meta (ускоряет restore-поиск) рядом с прокси.

    Пишется:
    - key, model_id, time, words_per_block, block_hashes, длина префикса в словах.
    """
    data = {
        "key": key,
        "model_id": model_id,
        "time": int(time.time()),
        "words_per_block": int(words_per_block),
        "block_hashes": list(block_hashes),
        "prefix_len_words": len(words_from_text(prefix_text)),
    }
    path = _meta_path_for_key(key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    log.info("meta_write key=%s blocks=%d wpb=%d path=%s", key[:16], len(block_hashes), words_per_block, path)

def scan_all_meta_local(limit: int = 200) -> List[Dict]:
    """
    Сканирование локального каталога .meta для поиска кандидатов restore.

    Возвращает:
    - Список метаданных (свежие первыми), ограниченный limit.
    """
    files = sorted(glob.glob(os.path.join(LOCAL_META_DIR, "*.meta.json")), key=os.path.getmtime, reverse=True)
    sel = files[:limit]
    out = []
    for p in sel:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            out.append(data)
        except Exception:
            continue
    log.debug("meta_scan files_total=%d selected=%d", len(files), len(sel))
    return out
