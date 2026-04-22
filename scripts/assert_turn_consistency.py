#!/usr/bin/env python3
"""
Audit runtime logs for race and speech regressions.

Checks:
- fallback ask followed by same agent question shortly after (race/dup candidate)
- long agent lines (word budget breach)
- duplicate question in same agent line
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class FallbackEvent:
    ts: datetime
    ask: str


def _parse_ts(line: str) -> datetime | None:
    # Supports both "2026-04-21 15:52:00,665" and "2026-04-21 15:52:00.665"
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]\d{3})", line)
    if not m:
        return None
    s = m.group(1).replace(",", ".")
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9?\s]", "", (s or "").lower())).strip()


def _word_count(s: str) -> int:
    return len([w for w in re.split(r"\s+", (s or "").strip()) if w])


def audit_log(path: str, *, race_window_sec: float, max_words: int) -> tuple[bool, dict[str, int]]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    fallback_events: list[FallbackEvent] = []
    race_dups = 0
    long_lines = 0
    dup_questions = 0
    agent_lines = 0

    for line in lines:
        ts = _parse_ts(line)
        if "line.session.ask_fallback" in line:
            m = re.search(r"Line fallback ask emitted:\s*(.+?)\"", line)
            ask = m.group(1).strip() if m else ""
            if ts and ask:
                fallback_events.append(FallbackEvent(ts=ts, ask=ask))
            continue

        if 'Agent said:' in line:
            agent_lines += 1
            m = re.search(r'Agent said:\s*"(.*)"', line)
            text = m.group(1) if m else ""
            if _word_count(text) > max_words:
                long_lines += 1

            qs = [_norm(q) for q in re.findall(r"[^?]+\?", text)]
            seen: set[str] = set()
            for q in qs:
                if q and q in seen:
                    dup_questions += 1
                    break
                seen.add(q)

            if ts and text and fallback_events:
                ntext = _norm(text)
                for fe in fallback_events:
                    dt = (ts - fe.ts).total_seconds()
                    if 0 <= dt <= race_window_sec:
                        if _norm(fe.ask) and _norm(fe.ask) in ntext:
                            race_dups += 1
                            break

    metrics = {
        "agent_lines": agent_lines,
        "race_dup_candidates": race_dups,
        "long_agent_lines": long_lines,
        "duplicate_questions": dup_questions,
    }
    ok = (race_dups == 0) and (long_lines == 0) and (dup_questions == 0)
    return ok, metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Audit voice logs for race/verbosity regressions")
    p.add_argument("--log-file", required=True, help="Path to runtime log (terminal capture)")
    p.add_argument("--race-window-sec", type=float, default=3.0, help="Window after fallback ask")
    p.add_argument("--max-words", type=int, default=12, help="Max words per agent line")
    args = p.parse_args()

    ok, metrics = audit_log(
        args.log_file,
        race_window_sec=args.race_window_sec,
        max_words=args.max_words,
    )
    print(metrics)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
