#!/usr/bin/env python
"""Render an episode trace as a single HTML page.

    python scripts/replay_trace.py runs/20260813-141200

Each step shows the frame the model saw, the body state it was given, what it thought,
and what it called -- which is how you tell a perception failure from a body-awareness
failure without re-running anything.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#12141a; color:#e6e8ee;
         font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:20px 28px; border-bottom:1px solid #262a35; background:#171a21; }}
  h1 {{ margin:0 0 6px; font-size:19px; letter-spacing:-.01em; }}
  .meta {{ color:#8b93a7; font-size:13px; }}
  .summary {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; }}
  .chip {{ background:#212632; border:1px solid #2e3442; border-radius:6px;
           padding:5px 11px; font-size:12px; }}
  .chip b {{ color:#fff; }}
  .ok {{ border-color:#2c6e4a; background:#16281f; }}
  .bad {{ border-color:#7a3540; background:#2a181c; }}
  main {{ padding:24px 28px; max-width:1500px; }}
  .step {{ display:grid; grid-template-columns:minmax(320px,460px) 1fr; gap:22px;
           padding:22px 0; border-bottom:1px solid #232734; }}
  .step img {{ width:100%; border-radius:8px; border:1px solid #2e3442; display:block; }}
  .n {{ color:#7dd3fc; font-weight:600; margin-bottom:8px; }}
  pre {{ background:#0e1016; border:1px solid #262a35; border-radius:7px; padding:11px 13px;
         white-space:pre-wrap; word-break:break-word; margin:0 0 12px; font-size:12.5px;
         font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .label {{ text-transform:uppercase; letter-spacing:.07em; font-size:10.5px;
            color:#8b93a7; margin:0 0 5px; }}
  .think {{ border-left:3px solid #7c6cf0; }}
  .call {{ border-left:3px solid #46a2ff; }}
  .res {{ border-left:3px solid #4ba97a; }}
  .err {{ border-left:3px solid #d9636f; }}
  .noimg {{ color:#5f6678; font-style:italic; padding:40px 0; text-align:center;
            border:1px dashed #2e3442; border-radius:8px; }}
</style>
<header>
  <h1>{task}</h1>
  <div class="meta">{model} &middot; {steps} steps &middot; {dirname}</div>
  <div class="summary">{chips}</div>
</header>
<main>{body}</main>
"""


def esc(value) -> str:
    return html.escape(str(value))


def build_chips(summary: dict) -> str:
    if not summary:
        return ""
    chips = []
    verified = summary.get("verified_success")
    claimed = summary.get("agent_claimed_success")
    chips.append(
        f'<span class="chip {"ok" if verified else "bad"}">verified: <b>{verified}</b></span>'
    )
    if claimed is not None and claimed != verified:
        chips.append(
            f'<span class="chip bad">agent <i>claimed</i>: <b>{claimed}</b> '
            f"&mdash; goal-detection failure</span>"
        )
    if summary.get("failure_kind"):
        chips.append(f'<span class="chip bad">failure: <b>{esc(summary["failure_kind"])}</b></span>')
    for key in ("tool_calls", "tool_errors", "redundant_tool_calls", "redundant_rate"):
        if key in summary:
            chips.append(f'<span class="chip">{key.replace("_", " ")}: <b>{summary[key]}</b></span>')
    if summary.get("verifier_verdict"):
        chips.append(f'<span class="chip">{esc(summary["verifier_verdict"])}</span>')
    return "".join(chips)


def build_step(record: dict) -> str:
    left = (
        f'<img src="{esc(record["frame"])}" alt="step {record["step"]}">'
        if record.get("frame")
        else '<div class="noimg">no new frame this step<br>(no world-changing action)</div>'
    )

    right = [f'<div class="n">step {record["step"]}</div>']
    if record.get("state_block"):
        right.append('<p class="label">body state given to the model</p>')
        right.append(f"<pre>{esc(record['state_block'])}</pre>")
    if record.get("thinking"):
        right.append('<p class="label">reasoning</p>')
        right.append(f'<pre class="think">{esc(record["thinking"])}</pre>')
    if record.get("text"):
        right.append('<p class="label">said</p>')
        right.append(f"<pre>{esc(record['text'])}</pre>")

    for call, result in zip(
        record.get("tool_calls", []), record.get("tool_results", []) + [""] * 8
    ):
        args = json.dumps(call.get("arguments", {}))
        right.append(f'<pre class="call">{esc(call["name"])}({esc(args)})</pre>')
        if result:
            cls = "err" if result.lower().startswith(("error", "move_to -> failed")) else "res"
            right.append(f'<pre class="{cls}">{esc(result)}</pre>')

    extra = record.get("tool_results", [])[len(record.get("tool_calls", [])) :]
    for result in extra:
        right.append(f'<pre class="res">{esc(result)}</pre>')

    return f'<div class="step"><div>{left}</div><div>{"".join(right)}</div></div>'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    trace_file = args.run_dir / "trace.jsonl"
    if not trace_file.exists():
        print(f"no trace.jsonl in {args.run_dir}")
        return 1

    records = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    summary_file = args.run_dir / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}

    page = PAGE.format(
        title=f"trace {args.run_dir.name}",
        task=esc(summary.get("task", "(unknown task)")),
        model=esc(summary.get("model", "?")),
        steps=len(records),
        dirname=esc(args.run_dir.name),
        chips=build_chips(summary),
        body="".join(build_step(r) for r in records),
    )

    output = args.output or args.run_dir / "replay.html"
    output.write_text(page)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
