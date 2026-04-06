"""Throughput benchmark: measures tok/s at varying concurrency.

Usage:
    python benchmarks/bench_throughput.py [--url URL] [--concurrencies 1,2,4,8,16]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp

DEFAULT_URL = "http://localhost:8014/v1/chat/completions"
DEFAULT_PROMPT = "Explain the physics of dropping a glass cup from 1 meter onto concrete. Be detailed."
DEFAULT_MAX_TOKENS = 512


async def single_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    req_id: int,
) -> tuple[int, int, float, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": 0.6,
    }
    start = time.monotonic()
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
    elapsed = time.monotonic() - start
    tokens = data["usage"]["completion_tokens"]
    return req_id, tokens, elapsed, tokens / elapsed if elapsed > 0 else 0


async def bench(url: str, model: str, n: int) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        # Warmup
        await single_request(session, url, model, -1)

        start = time.monotonic()
        tasks = [single_request(session, url, model, i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - start

    total_tok = sum(r[1] for r in results)
    throughput = total_tok / wall
    per_req = [r[3] for r in results]
    avg = sum(per_req) / len(per_req)

    return {
        "concurrency": n,
        "wall_time_s": round(wall, 2),
        "total_tokens": total_tok,
        "throughput_tok_s": round(throughput, 1),
        "per_request_avg_tok_s": round(avg, 1),
        "per_request_min_tok_s": round(min(per_req), 1),
        "per_request_max_tok_s": round(max(per_req), 1),
    }


async def main():
    parser = argparse.ArgumentParser(description="Trinity throughput benchmark")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--concurrencies", default="1,2,4,8,12,16", help="Comma-separated concurrency levels")
    parser.add_argument("--label", default="", help="Label for this run (e.g. 'turbo' or 'baseline')")
    args = parser.parse_args()

    # Discover model name
    # URL like http://localhost:8014/v1/chat/completions → base = http://localhost:8014
    api_base = args.url.split("/v1/")[0]
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{api_base}/v1/models") as resp:
            models_resp = await resp.json()
            model = models_resp["data"][0]["id"]

    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    label = args.label or "run"

    print(f"=== Trinity Throughput Benchmark [{label}] ===")
    print(f"Model: {model}")
    print(f"URL: {args.url}")
    print(f"Prompt tokens: ~22, Max output: {DEFAULT_MAX_TOKENS}")
    print()

    results = []
    for n in concurrencies:
        r = await bench(args.url, model, n)
        results.append(r)
        print(f"  {n:>2} parallel: {r['throughput_tok_s']:>8.1f} tok/s total, "
              f"{r['per_request_avg_tok_s']:>6.1f} tok/s/req, "
              f"wall={r['wall_time_s']:.1f}s")

    # Summary table
    print(f"\n=== Summary [{label}] ===")
    print(f"{'Parallel':>8} | {'Total tok/s':>12} | {'Per-req tok/s':>14} | {'Efficiency':>10}")
    print("-" * 55)
    base = results[0]["throughput_tok_s"]
    for r in results:
        eff = r["throughput_tok_s"] / base if base > 0 else 0
        print(f"{r['concurrency']:>8} | {r['throughput_tok_s']:>10.1f}  | "
              f"{r['per_request_avg_tok_s']:>12.1f}  | {eff:>8.2f}x")

    # Save JSON
    output = {
        "label": label,
        "model": model,
        "url": args.url,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "results": results,
    }
    fname = f"benchmarks/results_{label}_{int(time.time())}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    asyncio.run(main())
