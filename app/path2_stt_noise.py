"""Shared Path2 STT noise-final filter (bridge + orchestrator debug WS)."""

from __future__ import annotations

import os


def path2_stt_final_is_drop_noise(text: str) -> bool:
    if os.getenv("PATH2_STT_DROP_NOISE_FINALS", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    t = (text or "").strip().lower().rstrip(".,!?")
    return t in _NOISE_TOKENS


_NOISE_TOKENS = frozenset({
    "you", "uh", "um", "ah", "eh", "oh", "hm", "hmm", "mm",
    "mmm", "mhm", "uh huh", "uh-huh", "huh", "ha", "okay",
    "so", "like", "yeah", "right",
})
