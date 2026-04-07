"""Quick throughput benchmark for TQ4 vs FP8 comparison.

Measures tok/s at various concurrency levels with short outputs.
"""
import argparse
import asyncio
import json
import time

import aiohttp


async def do_request(session, url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
    elapsed = time.perf_counter() - t0
    if "error" in data:
        return {"error": data["error"], "elapsed": elapsed}
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    return {
        "completion_tokens": completion_tokens,
        "elapsed": elapsed,
        "tok_per_sec": completion_tokens / elapsed if elapsed > 0 else 0,
    }


async def bench_concurrency(url, model, concurrency, max_tokens, prompt):
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            do_request(session, url, model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    if not ok:
        return {"concurrency": concurrency, "error": "all failed", "errors": errors}

    total_tokens = sum(r["completion_tokens"] for r in ok)
    avg_per_req = sum(r["tok_per_sec"] for r in ok) / len(ok)

    return {
        "concurrency": concurrency,
        "num_ok": len(ok),
        "num_errors": len(errors),
        "wall_time": round(wall_time, 2),
        "total_tokens": total_tokens,
        "aggregate_tok_s": round(total_tokens / wall_time, 1),
        "avg_per_req_tok_s": round(avg_per_req, 1),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8014/v1/chat/completions")
    parser.add_argument("--model", default="/media/tonoken/CT4000/Models/arcee-ai/Trinity-Large-Thinking-W4A16")
    parser.add_argument("--concurrencies", default="1,4,8,16", help="Comma-separated")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Write a short poem about the ocean in exactly 4 lines.")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    concurrencies = [int(c) for c in args.concurrencies.split(",")]

    print(f"=== Benchmark: {args.label} ===")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    results = []
    for c in concurrencies:
        print(f"  Testing {c} concurrent requests...", end=" ", flush=True)
        r = await bench_concurrency(args.url, args.model, c, args.max_tokens, args.prompt)
        results.append(r)
        if "error" in r:
            print(f"FAILED: {r['error']}")
        else:
            print(f"{r['aggregate_tok_s']} tok/s aggregate, "
                  f"{r['avg_per_req_tok_s']} tok/s/req, "
                  f"{r['wall_time']}s wall")

    print()
    print(json.dumps({"label": args.label, "results": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
