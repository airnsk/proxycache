# slot_manager.py
# -*- coding: utf-8 -*-

"""
Менеджер слотов (SlotManager): глобально управляет KV-кэшами по всем бэкендам.

Назначение:
- Поддерживает стратегию выбора слота для «больших» запросов: active-exact → active-lcp (с порогом) → restore-lcp (с порогом) → cold.
- Обеспечивает REJECT для недостаточно похожих hot слотов, чтобы не перетирать полезный кэш.
- Реализует планировщик слотов для «малых» запросов: предпочитает свободный слот на «предпочитаемом» бэкенде,
  затем любые свободные, затем холодные (hot=False), затем самые старые по LRU.
- Сохраняет и восстанавливает KV-кэш через llama_client (basename = ключ префикса).
- Ведёт глобальные привязки SlotBinding и LRU, защищает слоты per-slot asyncio.Lock.

Глобальные идентификаторы:
- Слот представлен парой (backend_id, local_slot_id) — единое пространство слотов над кластером.

Логи:
- На каждом этапе выводит key=value метрики: prefer_backend, slot_select (free/cold/oldest), ensure_* (источник, LCP, ratio),
  cache_save/cache_restore и финальные события (release/mark_cold).
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

from fastapi import HTTPException

from llama_client import LlamaClient
from config import (
    DEFAULT_MODEL_ID,
    DISK_META_SCAN_LIMIT,
    PINNED_PREFIX_KEYS,
    SIMILARITY_MIN_RATIO,
    DEFAULT_WORDS_PER_BLOCK,
)
import hashing as hs

log = logging.getLogger(__name__)

# Глобальный тип слота: (backend_id, local_slot_id)
GSlot = Tuple[int, int]

@dataclass
class Backend:
    """
    Описание одного бэкенда llama.cpp.

    Поля:
    - id: индекс бэкенда в конфигурации.
    - url: адрес llama.cpp.
    - slots: число локальных слотов на сервере.
    - client: LlamaClient для этого бэкенда.
    """
    id: int
    url: str
    slots: int
    client: LlamaClient

@dataclass
class SlotBinding:
    """
    Привязка глобального слота к ключу префикса и его параметрам.

    Поля:
    - backend_id, local_slot_id: идентификация слота.
    - key: ключ префикса (SHA-256 канонического текста).
    - prefix_text: канонический текст (для .meta).
    - block_hashes: цепочка блок-хэшей для LCP.
    - words_per_block: размер блока слов.
    - hot: признак «горячего» слота (годен для reuse).
    - last_used_ts: отметка времени для LRU-управления.
    """
    backend_id: int
    local_slot_id: int
    key: str
    prefix_text: str
    block_hashes: List[str]
    words_per_block: int
    hot: bool = True
    last_used_ts: float = field(default_factory=lambda: time.time())

class SlotManager:
    """
    Глобальный менеджер слотов: хранит биндинги и локи, решает маршрутизацию и восстановление кэша.
    """

    def __init__(
        self,
        backends: List[Backend],
        model_id: str = DEFAULT_MODEL_ID,
        similarity_min_ratio: float = SIMILARITY_MIN_RATIO,
        words_per_block_default: int = DEFAULT_WORDS_PER_BLOCK,
    ):
        """
        Инициализация менеджера.

        Аргументы:
        - backends: список доступных бэкендов (url, slots, client).
        - model_id: идентификатор модели (для .meta).
        - similarity_min_ratio: порог похожести для LCP reuse/restore.
        - words_per_block_default: размер блоков по умолчанию.
        """
        self._backends: List[Backend] = backends
        self._model_id = model_id
        self._sim_ratio = similarity_min_ratio
        self._wpb_default = words_per_block_default

        # Все возможные gslots (декартово произведение backend x slots)
        self._all_slots: List[GSlot] = []
        for be in backends:
            for s in range(be.slots):
                self._all_slots.append((be.id, s))

        # Глобальные привязки и пер-слотовые замки
        self._bindings: Dict[GSlot, SlotBinding] = {}
        self._locks: Dict[GSlot, asyncio.Lock] = {g: asyncio.Lock() for g in self._all_slots}

        # Для LRU
        self._last_used: Dict[GSlot, float] = {}
        log.info("slot_manager_init backends=%d total_slots=%d", len(backends), len(self._all_slots))

    def _prefer_backend(self, req_key: str) -> int:
        """
        Простое распределение по хэшу ключа на id бэкенда (локальность).

        Возвращает:
        - backend_id для «предпочтительного» выбора.
        """
        h = int(req_key[:8], 16) if req_key and len(req_key) >= 8 else hash(req_key)
        idx = abs(h) % len(self._backends)
        log.debug("prefer_backend key=%s backend=%d", req_key[:16], idx)
        return idx

    def _slot_state(self, g: GSlot) -> str:
        """
        Человекочитаемое состояние слота: free/hot/cold.
        """
        b = self._bindings.get(g)
        if b is None:
            return "free"
        return "hot" if b.hot else "cold"

    def _free_slots_all(self, exclude: Set[GSlot]) -> List[GSlot]:
        """
        Список свободных gslot, исключая exclude.
        """
        out = []
        for g in self._all_slots:
            if g in exclude:
                continue
            if g not in self._bindings:
                out.append(g)
        return out

    def _lrud_slots(self, exclude: Set[GSlot]) -> List[GSlot]:
        """
        Занятые gslot по возрастанию last_used_ts (LRU первым), исключая exclude.
        """
        pairs = []
        for g, b in self._bindings.items():
            if g in exclude:
                continue
            ts = self._last_used.get(g, b.last_used_ts)
            pairs.append((ts, g))
        pairs.sort(key=lambda x: x[0])
        return [g for _, g in pairs]

    def _backend(self, backend_id: int) -> Backend:
        """
        Получить описание бэкенда по id.
        """
        return self._backends[backend_id]

    def _binding(self, g: GSlot) -> Optional[SlotBinding]:
        """
        Получить SlotBinding для gslot, если существует.
        """
        return self._bindings.get(g)

    async def acquire_free_or_cold_slot(
        self,
        exclude: Optional[Set[GSlot]] = None,
        prefer_backend_id: Optional[int] = None,
    ) -> Tuple[GSlot, asyncio.Lock]:
        """
        Выбрать слот под запрос (для «малых» и для cold/restore «больших»).

        Приоритет:
        1) Свободный слот на предпочитаемом бэкенде.
        2) Любой свободный слот.
        3) «Холодный» занятый (hot=False) по LRU и не pinned.
        4) Самый старый занятый по LRU (если не pinned).

        Возвращает:
        - (gslot, lock) — вызывающий обязан lock.acquire() перед использованием.
        """
        exclude = exclude or set()

        # 1) Свободный на предпочитаемом
        if prefer_backend_id is not None:
            for g in self._all_slots:
                if g in exclude:
                    continue
                if g[0] != prefer_backend_id:
                    continue
                if g not in self._bindings:
                    log.info("slot_select be=%d slot=%d state=%s reason=free_preferred", g[0], g[1], self._slot_state(g))
                    return g, self._locks[g]

        # 2) Любой свободный
        free_any = self._free_slots_all(exclude)
        if free_any:
            g = free_any[0]
            log.info("slot_select be=%d slot=%d state=%s reason=free_any", g[0], g[1], self._slot_state(g))
            return g, self._locks[g]

        # 3) Холодный (LRU)
        for g in self._lrud_slots(exclude):
            b = self._bindings[g]
            if not b.hot and b.key not in PINNED_PREFIX_KEYS:
                log.info("slot_select be=%d slot=%d state=%s reason=cold_lru", g[0], g[1], self._slot_state(g))
                return g, self._locks[g]

        # 4) Старый занятый (если не pinned)
        for g in self._lrud_slots(exclude):
            b = self._bindings[g]
            if b.key not in PINNED_PREFIX_KEYS:
                log.info("slot_select be=%d slot=%d state=%s reason=oldest_lru", g[0], g[1], self._slot_state(g))
                return g, self._locks[g]

        lrud = self._lrud_slots(exclude)
        if not lrud:
            log.error("slot_select_failed reason=no_slots")
            raise HTTPException(503, "No slots available")
        g = lrud[0]
        log.warning("slot_select be=%d slot=%d state=%s reason=wait_oldest_all_pinned", g[0], g[1], self._slot_state(g))
        return g, self._locks[g]

    def _best_active_exact(self, req_blocks: List[str]) -> Optional[Tuple[GSlot, SlotBinding]]:
        """
        Поиск точного совпадения среди hot привязок.
        """
        for g, b in self._bindings.items():
            if b.hot and b.block_hashes == req_blocks:
                return g, b
        return None

    def _best_active_lcp(self, req_blocks: List[str]) -> Optional[Tuple[GSlot, SlotBinding, int, float]]:
        """
        Поиск наилучшего LCP среди hot привязок.

        Возвращает:
        - (gslot, binding, lcp_count, ratio)
        """
        best = None
        for g, b in self._bindings.items():
            if not b.hot:
                continue
            l = hs.lcp_blocks(req_blocks, b.block_hashes)
            denom = max(1, min(len(req_blocks), len(b.block_hashes)))
            ratio = l / denom
            if best is None or ratio > best[3]:
                best = (g, b, l, ratio)
        return best

    def _best_restore_candidate(self, req_blocks: List[str], wpb: int) -> Optional[Tuple[str, int, float, List[str]]]:
        """
        Ищем лучший .meta кандидат по LCP.

        Возвращает:
        - (key, lcp, ratio, cand_blocks)
        """
        metas = hs.scan_all_meta_local(DISK_META_SCAN_LIMIT)
        best = None
        for m in metas:
            if int(m.get("words_per_block") or wpb) != wpb:
                continue
            cand_blocks = m.get("block_hashes") or []
            l = hs.lcp_blocks(req_blocks, cand_blocks)
            denom = max(1, min(len(req_blocks), len(cand_blocks)))
            ratio = l / denom
            if best is None or ratio > best[2]:
                best = (m.get("key"), l, ratio, cand_blocks)
        return best

    async def ensure_slot_for_request(
        self,
        req_key: str,
        prefix_text: str,
        req_blocks: List[str],
        words_per_block: int,
    ) -> Tuple[GSlot, SlotBinding, str, int, int]:
        """
        Назначить слот для «большого» запроса по стратегии active/restore/cold.

        Возвращает:
        - (gslot, binding, source, lcp_count, binding_total)

        Гарантии:
        - Возвращаемый слот уже имеет захваченный lock; вызывающий обязан release в finally.
        """
        exclude: Set[GSlot] = set()
        log.info("ensure_start key=%s req_blocks=%d wpb=%d bindings=%d", req_key[:16], len(req_blocks), words_per_block, len(self._bindings))

        # 1) exact среди hot
        ex = self._best_active_exact(req_blocks)
        if ex:
            g, b = ex
            lock = self._locks[g]
            await lock.acquire()
            self.touch(g)
            log.info("ensure_pick source=active-exact be=%d slot=%d", g[0], g[1])
            return g, b, "active-exact", len(req_blocks), len(self._bindings)

        # 2) active-lcp среди hot
        lcp_best = self._best_active_lcp(req_blocks)
        if lcp_best:
            g, b, lcp_cnt, ratio = lcp_best
            log.info("ensure_active_lcp be=%d slot=%d lcp=%d ratio=%.3f threshold=%.3f", g[0], g[1], lcp_cnt, ratio, self._sim_ratio)
            if ratio >= self._sim_ratio:
                lock = self._locks[g]
                await lock.acquire()
                self.touch(g)
                log.info("ensure_pick source=active-lcp be=%d slot=%d", g[0], g[1])
                return g, b, "active-lcp", lcp_cnt, len(self._bindings)
            else:
                exclude.add(g)
                log.info("ensure_reject be=%d slot=%d reason=ratio_below", g[0], g[1])

        # 3) restore-lcp по .meta
        cand = self._best_restore_candidate(req_blocks, words_per_block)
        if cand:
            key2, lcp_cnt, ratio, cand_blocks = cand
            log.info("ensure_restore_candidate key=%s lcp=%d ratio=%.3f threshold=%.3f", str(key2)[:16], lcp_cnt, ratio, self._sim_ratio)
            if ratio >= self._sim_ratio:
                prefer_be = self._prefer_backend(req_key)
                g, lock = await self.acquire_free_or_cold_slot(exclude=exclude, prefer_backend_id=prefer_be)
                await lock.acquire()
                await self.restore_slot_cache(g, key2)
                b = SlotBinding(
                    backend_id=g[0],
                    local_slot_id=g[1],
                    key=req_key,
                    prefix_text=prefix_text,
                    block_hashes=req_blocks,
                    words_per_block=words_per_block,
                    hot=True,
                )
                self._bindings[g] = b
                self.touch(g)
                log.info("ensure_pick source=restore-lcp be=%d slot=%d restore_key=%s", g[0], g[1], str(key2)[:16])
                return g, b, "restore-lcp", lcp_cnt, len(self._bindings)

        # 4) cold
        prefer_be = self._prefer_backend(req_key)
        g, lock = await self.acquire_free_or_cold_slot(exclude=exclude, prefer_backend_id=prefer_be)
        await lock.acquire()
        b = SlotBinding(
            backend_id=g[0],
            local_slot_id=g[1],
            key=req_key,
            prefix_text=prefix_text,
            block_hashes=req_blocks,
            words_per_block=words_per_block,
            hot=True,
        )
        self._bindings[g] = b
        self.touch(g)
        log.info("ensure_pick source=cold be=%d slot=%d", g[0], g[1])
        return g, b, "cold", 0, len(self._bindings)

    async def save_slot_cache(self, g: GSlot, key: str) -> None:
        """
        Сохранить кэш слота на сервере и записать локальную .meta.

        Аргументы:
        - g: глобальный слот (backend_id, slot_id).
        - key: basename/ключ префикса (без пути).
        """
        be = self._backend(g[0])
        log.info("cache_save be=%d slot=%d key=%s", g[0], g[1], key[:16])
        await be.client.save_slot(g[1], basename=key)
        b = self._bindings.get(g)
        if b:
            hs.write_meta_for_key_local(
                key=key,
                prefix_text=b.prefix_text,
                model_id=self._model_id,
                words_per_block=b.words_per_block,
                block_hashes=b.block_hashes,
            )

    async def restore_slot_cache(self, g: GSlot, key: str) -> None:
        """
        Восстановить кэш слота с сервера по basename = key.

        Аргументы:
        - g: глобальный слот.
        - key: basename/ключ префикса.
        """
        be = self._backend(g[0])
        log.info("cache_restore be=%d slot=%d key=%s", g[0], g[1], key[:16])
        await be.client.restore_slot(g[1], basename=key)

    def touch(self, g: GSlot) -> None:
        """
        Обновить метку времени LRU для gslot (важно во время длительного stream).
        """
        ts = time.time()
        self._last_used[g] = ts
        b = self._bindings.get(g)
        if b:
            b.last_used_ts = ts

    def release(self, g: GSlot) -> None:
        """
        Освободить lock слота (безопасно).
        """
        lock = self._locks[g]
        if lock.locked():
            lock.release()
            log.debug("slot_release be=%d slot=%d", g[0], g[1])

    def mark_slot_cold(self, g: GSlot) -> None:
        """
        Пометить занятый слот как cold (hot=False), пригоден для «малых» или эвикции.
        """
        b = self._bindings.get(g)
        if b and b.hot:
            b.hot = False
            log.info("slot_mark_cold be=%d slot=%d key=%s", g[0], g[1], b.key[:16])

    def get_binding(self, g: GSlot) -> Optional[SlotBinding]:
        """
        Вернуть SlotBinding для gslot, если существует.
        """
        return self._bindings.get(g)
