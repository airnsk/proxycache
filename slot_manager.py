# slot_manager.py

# -*- coding: utf-8 -*-

"""
SlotManager с учётом hot/cold слотов и обязательного сохранения
большого KV при их перезаписи.

- Слоты: (backend_id, local_slot_id).
- _get_free_or_oldest(): сначала свободный (ещё не использовался), иначе самый старый по времени.
- Для больших запросов:
    * если в выбранном слоте уже есть другой большой KV (big_key != restore_key),
      сначала обязательно сохраняем его на диск и обновляем мету;
    * затем, если есть restore_key, делаем restore.
- Сохранение после запроса делается только для больших запросов (is_big=True),
  и по успешному save слот помечается как HOT для этого key.
"""

import time
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional

from config import BACKENDS
import hashing as hs  # для touch_meta

log = logging.getLogger(__name__)

GSlot = Tuple[int, int]  # (backend_id, local_slot_id)


class SlotHeat(str, Enum):
    COLD = "cold"
    HOT = "hot"   # в слоте лежит большой KV для big_key


@dataclass
class SlotState:
    heat: SlotHeat = SlotHeat.COLD
    big_key: Optional[str] = None       # ключ большого KV, который мы считаем живущим в этом слоте
    last_used_at: float = 0.0           # для LRU
    last_saved_at: float = 0.0          # когда в последний раз делали save большого KV


class SlotManager:
    def __init__(self):
        self.backends: List[Dict] = []
        total_slots = 0

        for be_id, conf in enumerate(BACKENDS):
            n_slots = int(conf["n_slots"])
            self.backends.append({"id": be_id, "client": None, "n_slots": n_slots})
            total_slots += n_slots

        self._all_slots: List[GSlot] = [
            (be_id, s)
            for be_id, be in enumerate(self.backends)
            for s in range(be["n_slots"])
        ]

        # LRU по времени использования
        self._last_used: Dict[GSlot, float] = {g: 0.0 for g in self._all_slots}

        # Состояние "hot/cold" и закреплённый большой key
        self._state: Dict[GSlot, SlotState] = {
            g: SlotState() for g in self._all_slots
        }

        # Локи на слот
        self._locks: Dict[GSlot, asyncio.Lock] = {
            g: asyncio.Lock() for g in self._all_slots
        }

        log.info("slot_manager n_backends=%d total_slots=%d",
                 len(self.backends), total_slots)

    def set_clients(self, clients: List):
        for i, client in enumerate(clients):
            self.backends[i]["client"] = client

    # --- Вспомогательные методы ---

    def _is_free_cold(self, g: GSlot) -> bool:
        """
        Слот считается "свободным/холодным", если last_used_at == 0.0,
        то есть ещё вообще не использовался.
        """
        return self._last_used.get(g, 0.0) == 0.0

    def _get_free_or_oldest(self) -> Tuple[GSlot, asyncio.Lock]:
        """
        Базовый выбор слота:
        - сначала любые "свободные/холодные" (никогда не использовались),
        - иначе самый старый по last_used_at (LRU).
        """
        free = [g for g in self._all_slots if self._is_free_cold(g)]
        if free:
            g = free[0]
            return g, self._locks[g]

        g = sorted(self._all_slots, key=lambda x: self._last_used.get(x, 0.0))[0]
        return g, self._locks[g]

    async def _ensure_hot_saved_before_eviction(self, g: GSlot, incoming_big_key: Optional[str]) -> None:
        """
        Если в слоте лежит большой KV (heat=HOT, big_key != None) и мы
        собираемся использовать слот под другой большой контекст
        (incoming_big_key is None или incoming_big_key != big_key),
        то сначала обязательно сохраняем текущий big_key на диск и
        обновляем его мету (timestamp).
        """
        st = self._state[g]
        old_key = st.big_key

        # Нет старого большого KV — нечего сохранять
        if not old_key:
            return

        # Если мы собираемся продолжать тот же самый big_key, то eviction нет
        if incoming_big_key and incoming_big_key == old_key:
            return

        client = self.backends[g[0]]["client"]
        slot_id = g[1]

        log.info("evict_hot_before_use g=%s old_key=%s incoming=%s",
                 g, old_key[:16], (incoming_big_key[:16] if incoming_big_key else None))

        try:
            ok = await client.save_slot(slot_id, old_key)
        except Exception as e:
            log.warning("evict_save_fail g=%s key=%s: %s", g, old_key[:16], e)
            ok = False

        now = time.time()
        self._last_used[g] = now
        st.last_used_at = now
        if ok:
            st.last_saved_at = now
            # Обновляем timestamp в мета-файле, чтобы этот key считался свежим
            hs.touch_meta(old_key)
            log.info("evict_hot_saved g=%s key=%s", g, old_key[:16])
        else:
            log.warning("evict_hot_not_saved g=%s key=%s", g, old_key[:16])

    # --- Основной API ---

    async def acquire_for_request(
        self,
        *,
        is_big: bool,
        restore_key: Optional[str] = None,
    ) -> Tuple[GSlot, asyncio.Lock, Optional[bool]]:
        """
        Выбор слота под запрос.

        - Для любых запросов: выбираем free/oldest (как раньше).
        - Для больших запросов:
            * если в выбранном слоте есть другой большой KV, сначала его сохраняем;
            * если есть restore_key — делаем restore в выбранный слот.
        - Для малых запросов restore не делаем.
        """
        g, lock = self._get_free_or_oldest()
        await lock.acquire()

        st = self._state[g]
        restored: Optional[bool] = None

        # Обновляем "последнее использование" сразу при занятии
        now = time.time()
        self._last_used[g] = now
        st.last_used_at = now

        if is_big:
            # Перед использованием большого слота под новый большой контекст
            # обязательно сохраняем предыдущий большой KV (если он был).
            await self._ensure_hot_saved_before_eviction(g, restore_key)

            if restore_key:
                client = self.backends[g[0]]["client"]
                try:
                    restored = await client.restore_slot(g[1], restore_key)
                except Exception as e:
                    log.warning("restore_fail g=%s key=%s: %s",
                                g, restore_key[:16], e)
                    restored = False

                log.info("restore_before_chat g=%s key=%s ok=%s",
                         g, restore_key[:16], restored)

                if restored:
                    # Считаем, что в слоте теперь живёт большой KV для restore_key
                    st.heat = SlotHeat.HOT
                    st.big_key = restore_key
                    st.last_used_at = time.time()
                    self._last_used[g] = st.last_used_at
            else:
                # Большой запрос без restore: пока слот остаётся в том виде,
                # в каком был; после save_after мы пометим его HOT под новым key.
                log.info("big_no_restore g=%s", g)
        else:
            # Малый запрос: никаких restore, состояние HOT/COLD не меняем здесь.
            log.debug("small_request g=%s", g)

        return g, lock, restored

    async def save_after(self, g: GSlot, key: str, is_big: bool) -> bool:
        """
        Сохранение состояния слота после запроса.

        - Для малых запросов (is_big=False) просто обновляем last_used и выходим.
        - Для больших:
            * всегда делаем save_slot(g, key);
            * при успехе помечаем слот как HOT для этого key.
        """
        now = time.time()
        self._last_used[g] = now
        st = self._state[g]
        st.last_used_at = now

        if not is_big:
            # Малые запросы не сохраняем на диск
            return False

        client = self.backends[g[0]]["client"]
        slot_id = g[1]

        try:
            ok = await client.save_slot(slot_id, key)
        except Exception as e:
            log.warning("save_fail g=%s key=%s: %s", g, key[:16], e)
            ok = False

        if ok:
            st.heat = SlotHeat.HOT
            st.big_key = key
            st.last_saved_at = time.time()
            log.info("save_after_ok g=%s key=%s", g, key[:16])
        else:
            log.warning("save_after_failed g=%s key=%s", g, key[:16])

        return ok

    def release(self, g: GSlot):
        """
        Освобождение слота для следующего запроса.
        """
        lock = self._locks[g]
        if lock.locked():
            lock.release()
