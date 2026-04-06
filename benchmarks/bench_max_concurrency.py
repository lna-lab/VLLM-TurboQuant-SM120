"""Maximum concurrency benchmark: push parallel requests until OOM or failure.

Tests 16, 32, 48, 64 parallel requests and reports the maximum sustainable
concurrency with throughput numbers.

Usage:
    python benchmarks/bench_max_concurrency.py [--url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp

DEFAULT_URL = "http://localhost:8014/v1/chat/completions"
PROMPT = "Explain the physics of dropping a glass cup from 1 meter onto concrete. Be detailed."
MAX_TOKENS = 512


async def single_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    req_id: int,
) -> tuple[int, int, float, float, str | None]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
    }
    start = time.monotonic()
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                return req_id, 0, time.monotonic() - start, 0.0, f"HTTP {resp.status}: {text[:200]}"
            data = await resp.json()
        elapsed = time.monotonic() - start
        tokens = data["usage"]["completion_tokens"]
        return req_id, tokens, elapsed, tokens / elapsed if elapsed > 0 else 0, None
    except Exception as e:
        return req_id, 0, time.monotonic() - start, 0.0, str(e)[:200]


async def bench_concurrency(url: str, model: str, n: int) -> dict:
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warmup with single request
        _, _, _, _, err = await single_request(session, url, model, -1)
        if err:
            return {"concurrency": n, "status": "warmup_failed", "error": err}

        print(f"  Testing {n} parallel...", end=" ", flush=True)
        start = time.monotonic()
        tasks = [single_request(session, url, model, i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - start

    errors = [r for r in results if r[4] is not None]
    successes = [r for r in results if r[4] is None]

    if errors:
        # Some requests failed
        error_msgs = list(set(r[4] for r in errors))
        print(f"FAILED ({len(errors)}/{n} errors)")
        return {
            "concurrency": n,
            "status": "partial_failure",
            "successes": len(successes),
            "failures": len(errors),
            "error_samples": error_msgs[:3],
            "wall_time_s": round(wall, 2),
        }

    total_tok = sum(r[1] for r in successes)
    throughput = total_tok / wall
    per_req = [r[3] for r in successes]
    avg = sum(per_req) / len(per_req)

    print(f"OK — {throughput:.0f} tok/s total, {avg:.1f} tok/s/req, wall={wall:.1f}s")

    return {
        "concurrency": n,
        "status": "ok",
        "wall_time_s": round(wall, 2),
        "total_tokens": total_tok,
        "throughput_tok_s": round(throughput, 1),
        "per_request_avg_tok_s": round(avg, 1),
        "per_request_min_tok_s": round(min(per_req), 1),
        "per_request_max_tok_s": round(max(per_req), 1),
    }


async def main():
    parser = argparse.ArgumentParser(description="Maximum concurrency benchmark")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    api_base = args.url.split("/v1/")[0]
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{api_base}/v1/models") as resp:
            models_resp = await resp.json()
            model = models_resp["data"][0]["id"]

    concurrencies = [16, 32, 48, 64]

    print("=" * 70)
    print("  RTX PRO 6000 Blackwell × 4 — Maximum Concurrency Benchmark")
    print("  Model: Trinity-Large-Thinking-W4A16 (398B MoE, TP=4)")
    print("  Plugin: trinity-turbo Phase 1 (FP8 KV cache)")
    print(f"  Max output tokens: {MAX_TOKENS}")
    print("=" * 70)
    print()

    results = []
    max_ok = None
    for n in concurrencies:
        r = await bench_concurrency(args.url, model, n)
        results.append(r)
        if r["status"] == "ok":
            max_ok = r
        else:
            print(f"  → Stopping: {n} parallel failed")
            break

    # Final report
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"{'Parallel':>8} | {'Status':>8} | {'Total tok/s':>12} | {'Per-req tok/s':>14} | {'Wall':>6}")
    print("-" * 62)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['concurrency']:>8} | {'OK':>8} | {r['throughput_tok_s']:>10.1f}  | "
                  f"{r['per_request_avg_tok_s']:>12.1f}  | {r['wall_time_s']:>5.1f}s")
        else:
            print(f"{r['concurrency']:>8} | {'FAIL':>8} | {'—':>12} | {'—':>14} | {'—':>6}")

    if max_ok:
        print()
        print(f"  Maximum sustainable concurrency: {max_ok['concurrency']} parallel")
        print(f"  Peak throughput: {max_ok['throughput_tok_s']} tok/s")
        print(f"  Per-request latency: {max_ok['per_request_avg_tok_s']} tok/s")
        print()
        print(f"  Hardware: 4× RTX PRO 6000 Blackwell (96GB each, SM120)")
        print(f"  Model: 398B MoE (13B active), W4A16 quantized")
        print(f"  KV cache: FP8, max context 256K")

    # Save results
    output = {
        "hardware": "4x RTX PRO 6000 Blackwell (96GB, SM120)",
        "model": model,
        "plugin": "trinity-turbo Phase 1",
        "kv_cache_dtype": "fp8_e4m3",
        "max_tokens": MAX_TOKENS,
        "gpu_memory_utilization": 0.95,
        "max_sustainable_concurrency": max_ok["concurrency"] if max_ok else 0,
        "peak_throughput_tok_s": max_ok["throughput_tok_s"] if max_ok else 0,
        "results": results,
    }
    fname = f"benchmarks/results_max_concurrency_{int(time.time())}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {fname}")


if __name__ == "__main__":
    asyncio.run(main())
