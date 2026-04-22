#!/usr/bin/env python3
"""
Soak test for replay-driven voice E2E.

Runs replay suite repeatedly and reports:
- pass rate
- p50/p95/max TTFT
- top failure messages
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from statistics import median

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.voice_agent_e2e_test import run_replays  # noqa: E402


def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[int(max(0, round(0.95 * (len(s) - 1))))]


async def _run(
    ws_url: str,
    replay_file: str,
    runs: int,
    post_user_idle: float,
    ttft_budget_ms: float,
) -> dict:
    all_lat: list[float] = []
    failures: Counter[str] = Counter()
    pass_count = 0

    for i in range(1, runs + 1):
        result = await run_replays(
            ws_url,
            replay_file=replay_file,
            post_user_idle=post_user_idle,
            ttft_budget_ms=ttft_budget_ms,
        )
        all_lat.extend(result.get("latency_ms", []))
        if result.get("ok"):
            pass_count += 1
        else:
            for err in result.get("errors", []):
                failures[err] += 1
        print(f"[run {i}/{runs}] {'OK' if result.get('ok') else 'FAIL'}")

    summary = {
        "runs": runs,
        "passed_runs": pass_count,
        "failed_runs": runs - pass_count,
        "pass_rate": (pass_count / runs) if runs else 0.0,
        "ttft_ms": {
            "n": len(all_lat),
            "p50": median(all_lat) if all_lat else 0.0,
            "p95": _p95(all_lat),
            "max": max(all_lat) if all_lat else 0.0,
        },
        "top_failures": failures.most_common(10),
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Replay soak test")
    p.add_argument("--ws-url", default=os.getenv("VOICE_E2E_WS", "ws://127.0.0.1:8000/ws"))
    p.add_argument(
        "--replay-file",
        default=os.getenv(
            "VOICE_E2E_REPLAY_FILE",
            os.path.join(_ROOT, "scripts", "fixtures", "voice_replays", "deep_regressions.json"),
        ),
    )
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--post-user-idle", type=float, default=35.0)
    p.add_argument("--ttft-budget-ms", type=float, default=1500.0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    t0 = time.monotonic()
    summary = asyncio.run(
        _run(
            args.ws_url,
            replay_file=args.replay_file,
            runs=args.runs,
            post_user_idle=args.post_user_idle,
            ttft_budget_ms=args.ttft_budget_ms,
        )
    )
    summary["elapsed_sec"] = round(time.monotonic() - t0, 2)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(summary)

    # Fail soak if any run failed.
    sys.exit(0 if summary["failed_runs"] == 0 else 1)


if __name__ == "__main__":
    main()
