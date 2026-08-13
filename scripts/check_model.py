#!/usr/bin/env python
"""Preflight: does this model, on this provider, actually do what the loop needs?

Vision-capable models on HF Inference are served by many different providers, and
function-calling support varies between them. Rather than discover that mid-episode, we
probe three capabilities up front and print the mode to run in.

    python scripts/check_model.py
    python scripts/check_model.py --model Qwen/Qwen3-VL-8B-Instruct --provider novita
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embodied_agent.perception.render import to_data_uri  # noqa: E402
from embodied_agent.reasoner.hf_chat import ACTION_SCHEMA, HF_ROUTER_BASE_URL  # noqa: E402

PASS, FAIL, WARN = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"

PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "report_colour",
            "description": "Report the colour of the square in the image.",
            "parameters": {
                "type": "object",
                "properties": {"colour": {"type": "string"}},
                "required": ["colour"],
            },
        },
    }
]


def probe_image() -> str:
    """A magenta square on white -- an unusual colour, so a guess is unlikely to pass."""
    canvas = np.full((160, 160, 3), 255, dtype=np.uint8)
    canvas[40:120, 40:120] = (220, 30, 190)
    return to_data_uri(canvas)


def make_client(args):
    from openai import OpenAI

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(f"{FAIL}  HF_TOKEN is not set. Put it in .env or export it.")
        sys.exit(1)
    url = args.base_url or HF_ROUTER_BASE_URL
    if args.provider:
        url = f"https://router.huggingface.co/{args.provider}/v1"
    return OpenAI(base_url=url, api_key=token, timeout=120.0), url


def check_vision(client, model) -> bool:
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What colour is the square? Answer with one word.",
                        },
                        {"type": "image_url", "image_url": {"url": probe_image()}},
                    ],
                }
            ],
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        ok = any(word in answer for word in ("magenta", "pink", "purple", "violet", "fuchsia"))
        print(f"{PASS if ok else WARN}  vision: model replied {answer[:60]!r}")
        if not ok:
            print("       (it answered, but not with the expected colour -- check the model)")
        return True
    except Exception as exc:
        print(f"{FAIL}  vision: {type(exc).__name__}: {exc}")
        return False


def check_tools(client, model) -> bool:
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=128,
            tools=PROBE_TOOL,
            tool_choice="auto",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Use report_colour to report the square's colour."},
                        {"type": "image_url", "image_url": {"url": probe_image()}},
                    ],
                }
            ],
        )
        calls = resp.choices[0].message.tool_calls or []
        if calls:
            print(f"{PASS}  tools: got tool_calls -> {calls[0].function.name}"
                  f"({calls[0].function.arguments[:60]})")
            return True
        print(f"{WARN}  tools: no tool_calls returned (model answered in prose instead)")
        return False
    except Exception as exc:
        print(f"{FAIL}  tools: {type(exc).__name__}: {exc}")
        return False


def check_json_schema(client, model) -> bool:
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=256,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "agent_step", "schema": ACTION_SCHEMA, "strict": True},
            },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Call the tool 'report_colour' with the square's colour.",
                        },
                        {"type": "image_url", "image_url": {"url": probe_image()}},
                    ],
                }
            ],
        )
        import json

        json.loads(resp.choices[0].message.content or "")
        print(f"{PASS}  json_schema: returned parseable structured output")
        return True
    except Exception as exc:
        print(f"{WARN}  json_schema: {type(exc).__name__}: {str(exc)[:150]}")
        return False


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--provider", default=os.environ.get("HF_PROVIDER"))
    parser.add_argument("--base-url", default=os.environ.get("HF_BASE_URL"))
    args = parser.parse_args()

    client, url = make_client(args)
    print(f"model:    {args.model}")
    print(f"endpoint: {url}\n")

    vision = check_vision(client, args.model)
    tools = check_tools(client, args.model)
    schema = check_json_schema(client, args.model)

    print()
    if not vision:
        print("This model cannot be used: the loop needs image input.")
        return 1
    if tools:
        print("Recommended: run with the default 'tools' mode.")
    elif schema:
        print("Recommended: run with --json-mode (no native tool calling on this provider).")
    else:
        print("Neither tool calling nor structured output worked. Try another provider "
              "with --provider, or another model.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
