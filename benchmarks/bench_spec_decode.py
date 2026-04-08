"""Speculative decoding A/B benchmark.

Measures latency, throughput, and output quality with and without spec decode.
Run against vLLM OpenAI-compatible API.

Usage:
    python benchmarks/bench_spec_decode.py --url http://localhost:8014 --label baseline
    python benchmarks/bench_spec_decode.py --url http://localhost:8014 --label spec_decode
"""
import argparse
import asyncio
import json
import time

import aiohttp


PROMPTS = [
    # Short factual
    "What is the capital of France? Answer in one sentence.",
    # Medium reasoning
    "Explain why the sky is blue in exactly 3 sentences.",
    # Longer generation
    "Write a short poem about the ocean in exactly 4 lines.",
    # Code
    "Write a Python function that computes the fibonacci sequence up to n. Include a docstring.",
    # Technical
    "Explain the difference between TCP and UDP in 5 bullet points.",
    # Creative
    "Describe a sunset over the mountains in 3 vivid sentences.",
]


async def do_request(session, url, model, prompt, max_tokens, temperature=0.0):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
    elapsed = time.perf_counter() - t0
    if "error" in data:
        return {"error": str(data["error"]), "elapsed": elapsed, "prompt": prompt}
    usage = data.get("usage", {})
    content = data["choices"][0]["message"]["content"] if data.get("choices") else ""
    return {
        "prompt": prompt,
        "completion": content,
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed": elapsed,
        "tok_per_sec": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0,
    }


async def bench_sequential(url, model, max_tokens):
    """Sequential requests — measures per-request latency."""
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = []
        for prompt in PROMPTS:
            r = await do_request(session, url, model, prompt, max_tokens)
            results.append(r)
    return results


async def bench_concurrent(url, model, concurrency, max_tokens):
    """Concurrent requests — measures throughput."""
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Use the same prompt for all concurrent requests for fair comparison
        prompt = "Write a detailed explanation of how neural networks learn, covering backpropagation, gradient descent, and loss functions. Be thorough."
        tasks = [
            do_request(session, url, model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"error": "all failed", "concurrency": concurrency}

    total_tokens = sum(r["completion_tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "wall_time": round(wall_time, 2),
        "total_tokens": total_tokens,
        "aggregate_tok_s": round(total_tokens / wall_time, 1),
        "avg_per_req_tok_s": round(sum(r["tok_per_sec"] for r in ok) / len(ok), 1),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8014/v1/chat/completions")
    parser.add_argument("--model", default="/media/tonoken/CT4000/Models/arcee-ai/Trinity-Large-Thinking-W4A16")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--concurrencies", default="1,4,8,16", help="Comma-separated")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Speculative Decoding Benchmark — {args.label}")
    print(f"{'='*60}")
    print(f"URL: {args.url}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Phase 1: Sequential (latency + quality)
    print("--- Phase 1: Sequential requests (latency + quality) ---")
    seq_results = await bench_sequential(args.url, args.model, args.max_tokens)

    for r in seq_results:
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  {r['tok_per_sec']:6.1f} tok/s  {r['completion_tokens']:3d} tok  {r['elapsed']:5.2f}s  | {r['prompt'][:50]}...")
            # Print first line of completion for quality check
            first_line = r['completion'].split('\n')[0][:80]
            print(f"    -> {first_line}")

    ok_seq = [r for r in seq_results if "error" not in r]
    if ok_seq:
        avg_lat = sum(r["elapsed"] for r in ok_seq) / len(ok_seq)
        avg_tps = sum(r["tok_per_sec"] for r in ok_seq) / len(ok_seq)
        print(f"\n  Average: {avg_tps:.1f} tok/s, {avg_lat:.2f}s latency")

    # Phase 2: Concurrent (throughput)
    print(f"\n--- Phase 2: Concurrent requests (throughput) ---")
    concurrencies = [int(c) for c in args.concurrencies.split(",")]
    conc_results = []
    for c in concurrencies:
        print(f"  {c} concurrent...", end=" ", flush=True)
        r = await bench_concurrent(args.url, args.model, c, args.max_tokens)
        conc_results.append(r)
        if "error" in r:
            print(f"FAILED: {r['error']}")
        else:
            print(f"{r['aggregate_tok_s']} tok/s agg, {r['avg_per_req_tok_s']} tok/s/req, {r['wall_time']}s")

    # Save results
    output = {
        "label": args.label,
        "max_tokens": args.max_tokens,
        "sequential": seq_results,
        "concurrent": conc_results,
    }
    outfile = f"benchmarks/results_spec_decode_{args.label}_{int(time.time())}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    asyncio.run(main())
