#!/usr/bin/env python3
"""
Replay-driven E2E checks against running Line /ws server.

Why:
- Unit tests pass but live conversations fail from timing/race/verbosity behavior.
- This script runs realistic multi-turn replays and enforces turn-level quality gates.

Usage:
  python scripts/voice_agent_e2e_test.py
  python scripts/voice_agent_e2e_test.py --replay-file scripts/fixtures/voice_replays/deep_regressions.json
  python scripts/voice_agent_e2e_test.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.response_lint import contains_address_leak  # noqa: E402


@dataclass
class TurnCase:
    user: str
    expect_any: list[str]
    expect_all: list[str]
    forbid_contains: list[str]
    max_words: int | None
    disallow_address_leak: bool


@dataclass
class ReplayCase:
    name: str
    address_tokens: list[str]
    turns: list[TurnCase]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9?\s]", "", s.lower())).strip()


def _extract_questions(text: str) -> list[str]:
    return [_norm(m) for m in re.findall(r"[^?]+\?", text or "")]


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def load_cases(path: str) -> list[ReplayCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[ReplayCase] = []
    for c in raw.get("cases", []):
        turns = []
        for t in c.get("turns", []):
            turns.append(
                TurnCase(
                    user=t.get("user", ""),
                    expect_any=list(t.get("expect_any", [])),
                    expect_all=list(t.get("expect_all", [])),
                    forbid_contains=list(t.get("forbid_contains", [])),
                    max_words=t.get("max_words"),
                    disallow_address_leak=bool(t.get("disallow_address_leak", False)),
                )
            )
        cases.append(
            ReplayCase(
                name=c.get("name", "unnamed"),
                address_tokens=list(c.get("address_tokens", [])),
                turns=turns,
            )
        )
    return cases


async def _recv_agent_turn(
    ws: Any,
    *,
    max_wait_sec: float,
    idle_gap_sec: float = 2.0,
) -> tuple[list[dict[str, Any]], float | None]:
    buf: list[dict[str, Any]] = []
    first_msg_at: float | None = None
    deadline = time.monotonic() + max_wait_sec
    last_rx = time.monotonic()

    while time.monotonic() < deadline:
        remaining = min(0.25, deadline - time.monotonic())
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            if time.monotonic() - last_rx >= idle_gap_sec and buf:
                break
            continue
        last_rx = time.monotonic()
        if isinstance(raw, bytes):
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        buf.append(msg)
        if first_msg_at is None and msg.get("type") == "message" and msg.get("content"):
            first_msg_at = time.monotonic()

    return buf, first_msg_at


async def _one_user_turn(ws: Any, text: str) -> None:
    for frame in (
        {"type": "user_state", "value": "speaking"},
        {"type": "message", "content": text},
        {"type": "user_state", "value": "idle"},
    ):
        await ws.send(json.dumps(frame))


def _concat_agent_messages(msgs: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in msgs:
        if m.get("type") == "message" and m.get("content"):
            parts.append(str(m["content"]))
    return "".join(parts)


def _validate_turn(case: ReplayCase, idx: int, tc: TurnCase, agent_text: str) -> list[str]:
    errs: list[str] = []
    t = _norm(agent_text)

    if tc.expect_any:
        if not any(_norm(x) in t for x in tc.expect_any):
            errs.append(f"turn{idx}: missing expect_any {tc.expect_any}")
    for must in tc.expect_all:
        if _norm(must) not in t:
            errs.append(f"turn{idx}: missing expect_all '{must}'")
    for bad in tc.forbid_contains:
        if _norm(bad) in t:
            errs.append(f"turn{idx}: contains forbidden '{bad}'")
    if tc.max_words is not None:
        wc = _word_count(agent_text)
        if wc > tc.max_words:
            errs.append(f"turn{idx}: too long {wc}>{tc.max_words} words")
    if tc.disallow_address_leak and contains_address_leak(agent_text, case.address_tokens):
        errs.append(f"turn{idx}: address leak in agent speech")

    qs = [q for q in _extract_questions(agent_text) if q]
    seen: set[str] = set()
    for q in qs:
        if q in seen:
            errs.append(f"turn{idx}: duplicate question in same turn: '{q}'")
            break
        seen.add(q)
    return errs


async def run_case(
    ws_url: str,
    *,
    case: ReplayCase,
    call_id: str,
    post_user_idle: float,
    ttft_budget_ms: float,
) -> dict[str, Any]:
    import websockets

    res: dict[str, Any] = {
        "case": case.name,
        "ok": True,
        "errors": [],
        "turns": [],
        "latency_ms": [],
    }
    async with websockets.connect(
        ws_url,
        ping_interval=20,
        ping_timeout=20,
        max_size=2**20,
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "call_id": call_id,
                    "agent_call_id": call_id,
                    "from": "+15555550100",
                    "to": "+15555550101",
                    "agent": {},
                    "metadata": {"source": "voice_agent_e2e_replay"},
                }
            )
        )

        await _recv_agent_turn(ws, max_wait_sec=15.0, idle_gap_sec=1.0)
        for i, turn in enumerate(case.turns, start=1):
            t_send = time.monotonic()
            await _one_user_turn(ws, turn.user)
            msgs, first_at = await _recv_agent_turn(
                ws,
                max_wait_sec=post_user_idle,
                idle_gap_sec=2.0,
            )
            agent_text = _concat_agent_messages(msgs)
            first_ms = (first_at - t_send) * 1000 if first_at else None
            if first_ms is not None:
                res["latency_ms"].append(first_ms)
                if first_ms > ttft_budget_ms:
                    res["errors"].append(
                        f"turn{i}: TTFT {first_ms:.0f}ms exceeds budget {ttft_budget_ms:.0f}ms"
                    )

            turn_errs = _validate_turn(case, i, turn, agent_text)
            res["errors"].extend(turn_errs)
            res["turns"].append(
                {
                    "idx": i,
                    "user": turn.user,
                    "agent": agent_text,
                    "first_ms": first_ms,
                    "errors": turn_errs,
                }
            )

    res["ok"] = len(res["errors"]) == 0
    return res


async def run_replays(
    ws_url: str,
    *,
    replay_file: str,
    post_user_idle: float,
    ttft_budget_ms: float,
) -> dict[str, Any]:
    cases = load_cases(replay_file)
    out: dict[str, Any] = {"ok": True, "cases": [], "errors": [], "latency_ms": []}
    for i, c in enumerate(cases, start=1):
        r = await run_case(
            ws_url,
            case=c,
            call_id=f"e2e-replay-{int(time.time())}-{i}",
            post_user_idle=post_user_idle,
            ttft_budget_ms=ttft_budget_ms,
        )
        out["cases"].append(r)
        out["latency_ms"].extend(r["latency_ms"])
        if not r["ok"]:
            out["ok"] = False
            out["errors"].extend([f"{c.name}: {e}" for e in r["errors"]])
    if out["latency_ms"]:
        vals = sorted(out["latency_ms"])
        p95 = vals[int(max(0, round(0.95 * (len(vals) - 1))))]
        out["latency_summary"] = {
            "n": len(vals),
            "p50_ms": median(vals),
            "p95_ms": p95,
            "max_ms": vals[-1],
        }
    else:
        out["latency_summary"] = {"n": 0}
    return out


def _default_replay_file() -> str:
    return os.path.join(
        _ROOT,
        "scripts",
        "fixtures",
        "voice_replays",
        "deep_regressions.json",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Replay-driven Line /ws E2E checks")
    p.add_argument(
        "--ws-url",
        default=os.getenv("VOICE_E2E_WS", "ws://127.0.0.1:8000/ws"),
        help="Line VoiceAgentApp WebSocket URL",
    )
    p.add_argument(
        "--replay-file",
        default=os.getenv("VOICE_E2E_REPLAY_FILE", _default_replay_file()),
        help="JSON replay fixture file",
    )
    p.add_argument(
        "--post-user-idle",
        type=float,
        default=float(os.getenv("VOICE_E2E_POST_USER_IDLE", "35")),
        help="Max seconds to wait for agent reply after each turn",
    )
    p.add_argument(
        "--ttft-budget-ms",
        type=float,
        default=float(os.getenv("VOICE_E2E_TTFT_BUDGET_MS", "1500")),
        help="Fail turn when time-to-first-text exceeds this budget",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result only",
    )
    args = p.parse_args()

    try:
        result = asyncio.run(
            run_replays(
                args.ws_url,
                replay_file=args.replay_file,
                post_user_idle=args.post_user_idle,
                ttft_budget_ms=args.ttft_budget_ms,
            )
        )
    except ConnectionRefusedError:
        print("FAIL: cannot connect — start server: python run.py", file=sys.stderr)
        sys.exit(2)
    except OSError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Replay file: {args.replay_file}")
        print(f"Overall: {'OK' if result['ok'] else 'FAIL'}")
        print(f"Latency: {result.get('latency_summary', {})}")
        if result["errors"]:
            for e in result["errors"]:
                print(f"  - {e}")

    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
