"""Long-context concurrency benchmark: ~80K input × N parallel.

Trinity-Large-Thinking uses <think> blocks that consume significant output tokens.
Input 80K + thinking + completion must fit within 262K max context.

Tests whether TQ4 KV compression enables more concurrent long-context requests.
vLLM: Maximum concurrency for 262,144 tokens: 25.64x (TQ4) vs ~18x (FP8).

Usage:
    python benchmarks/bench_long_context.py --concurrency 24 --context-tokens 80000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp

_FILLER = (
    "The quick brown fox jumps over the lazy dog near the river bank. "
    "A gentle breeze carried the scent of pine trees across the meadow. "
    "Stars twinkled above as the night grew darker and colder. "
    "The ancient library held thousands of books from every era. "
)


def make_long_prompt(target_tokens: int) -> str:
    chars_needed = target_tokens * 4
    repeats = chars_needed // len(_FILLER) + 1
    text = _FILLER * repeats
    text = text[:chars_needed] + "\n\nSummarize the above passage in exactly 4 sentences."
    return text


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    req_id: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.monotonic()
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"id": req_id, "error": f"HTTP {resp.status}: {text[:300]}",
                        "elapsed": time.monotonic() - t0}
            data = await resp.json()
        elapsed = time.monotonic() - t0
        usage = data.get("usage", {})
        return {
            "id": req_id,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "elapsed": round(elapsed, 2),
            "decode_tok_s": round(usage.get("completion_tokens", 0) / elapsed, 1) if elapsed > 0 else 0,
        }
    except Exception as e:
        return {"id": req_id, "error": str(e)[:300], "elapsed": time.monotonic() - t0}


async def main():
    parser = argparse.ArgumentParser(description="Long-context concurrency benchmark")
    parser.add_argument("--url", default="http://localhost:8014/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--context-tokens", type=int, default=80000,
                        help="Approximate input tokens per request (80K default for thinking model)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max output tokens (thinking + completion)")
    parser.add_argument("--label", default="long_context")
    args = parser.parse_args()

    api_base = args.url.split("/v1/")[0]
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{api_base}/v1/models") as resp:
            model = (await resp.json())["data"][0]["id"]

    prompt = make_long_prompt(args.context_tokens)

    print(f"=== Long-Context Benchmark: {args.label} ===")
    print(f"Model: {model.split('/')[-1]}")
    print(f"Target input: ~{args.context_tokens:,} tokens ({len(prompt):,} chars)")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max output: {args.max_tokens} tokens")
    print(f"Headroom per req: 262K - 80K - {args.max_tokens} = ~{262144 - args.context_tokens - args.max_tokens:,} tokens for <think>")
    print()

    # Warmup
    print("Warmup...", end=" ", flush=True)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        warmup = await send_request(session, args.url, model, "Hi", 8, -1)
        if "error" in warmup:
            print(f"FAILED: {warmup['error']}")
            return
        print("OK")

        print(f"Sending {args.concurrency} × ~{args.context_tokens//1000}K requests...", flush=True)
        t0 = time.monotonic()
        tasks = [
            send_request(session, args.url, model, prompt, args.max_tokens, i)
            for i in range(args.concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.monotonic() - t0

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    print()
    print(f"=== Results ===")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"Success: {len(ok)}/{args.concurrency}")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"\nError samples:")
        for e in errors[:3]:
            print(f"  req {e['id']}: {e['error'][:200]}")

    if ok:
        total_prompt = sum(r["prompt_tokens"] for r in ok)
        total_completion = sum(r["completion_tokens"] for r in ok)
        avg_prompt = total_prompt / len(ok)
        avg_decode = sum(r["decode_tok_s"] for r in ok) / len(ok)
        agg_decode = total_completion / wall_time

        print(f"\nAvg prompt tokens: {avg_prompt:,.0f}")
        print(f"Total completion tokens: {total_completion:,}")
        print(f"Aggregate decode: {agg_decode:.1f} tok/s")
        print(f"Per-request decode: {avg_decode:.1f} tok/s")

        print(f"\nPer-request breakdown:")
        print(f"{'ID':>4} | {'Prompt':>8} | {'Compl':>6} | {'Time':>7} | {'tok/s':>7}")
        print("-" * 45)
        for r in sorted(ok, key=lambda x: x["id"]):
            print(f"{r['id']:>4} | {r['prompt_tokens']:>8,} | {r['completion_tokens']:>6} | "
                  f"{r['elapsed']:>6.1f}s | {r['decode_tok_s']:>6.1f}")

    output = {
        "label": args.label,
        "model": model,
        "target_context_tokens": args.context_tokens,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "wall_time": round(wall_time, 2),
        "num_ok": len(ok),
        "num_errors": len(errors),
        "results": results,
    }
    fname = f"benchmarks/results_{args.label}_{int(time.time())}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {fname}")


if __name__ == "__main__":
    asyncio.run(main())
