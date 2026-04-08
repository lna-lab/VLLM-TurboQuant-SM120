"""Fleet benchmark: 8-worker pipelined task processing.

Demonstrates the 8-parallel pipeline pattern — 2 seq/GPU sweet spot.
Submits N tasks to the Fleet and measures throughput + utilization.

Usage:
    python benchmarks/bench_fleet.py --tasks 24 --workers 8
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time

from trinity_turbo.fleet import Fleet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Sample tasks: diverse prompts to avoid prefix cache skewing results
TASK_PROMPTS = [
    "Explain the theory of general relativity and its implications for GPS satellites.",
    "Compare and contrast TCP and UDP protocols. When would you use each?",
    "Describe the process of photosynthesis in detail, including the light and dark reactions.",
    "What are the main differences between Python and Rust? Give code examples.",
    "Explain how transformers work in deep learning, focusing on the attention mechanism.",
    "Describe the history and cultural significance of sushi in Japanese cuisine.",
    "What causes ocean tides? Explain the roles of the Moon and Sun.",
    "Compare functional programming and object-oriented programming paradigms.",
    "Explain quantum entanglement in terms a high school student would understand.",
    "Describe the water cycle and its importance to Earth's climate system.",
    "What is CRISPR and how is it used in gene editing? What are the ethical concerns?",
    "Explain the concept of blockchain and how it ensures data integrity.",
    "Describe the life cycle of a star, from nebula to its final state.",
    "What are microservices? Compare them to monolithic architecture.",
    "Explain how vaccines work and the difference between mRNA and traditional vaccines.",
    "Describe the principles behind nuclear fusion and why it's difficult to achieve.",
    "What is the Turing test and why is it significant in AI research?",
    "Explain the economics of supply and demand with real-world examples.",
    "Describe the architecture of a modern CPU, including caches and pipelines.",
    "What causes earthquakes? Explain plate tectonics and seismic waves.",
    "Compare different sorting algorithms and their time complexities.",
    "Explain the greenhouse effect and its relationship to climate change.",
    "Describe how the human immune system responds to a viral infection.",
    "What is dark matter and dark energy? What evidence supports their existence?",
    "Explain the CAP theorem in distributed systems with practical examples.",
    "Describe the history of the Internet, from ARPANET to the modern web.",
    "What is the Standard Model of particle physics? What particles does it describe?",
    "Explain database indexing and how B-trees improve query performance.",
    "Describe the formation of the solar system according to current scientific models.",
    "What is reinforcement learning? Explain Q-learning with a simple example.",
    "Explain the principles of aerodynamics and how airplanes generate lift.",
    "Describe the process of brewing beer, from malting to fermentation.",
]


async def main():
    parser = argparse.ArgumentParser(description="Fleet benchmark")
    parser.add_argument("--url", default="http://localhost:8014/v1/chat/completions")
    parser.add_argument("--tasks", type=int, default=24, help="Number of tasks to process")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers (default: 8 = sweet spot)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per response")
    args = parser.parse_args()

    fleet = Fleet(url=args.url, workers=args.workers, timeout=600)

    # Submit tasks
    for i in range(args.tasks):
        prompt = TASK_PROMPTS[i % len(TASK_PROMPTS)]
        fleet.submit(prompt, task_id=f"task_{i:03d}", max_tokens=args.max_tokens)

    print(f"=== Fleet Benchmark: {args.tasks} tasks × {args.workers} workers ===")
    print(f"Pattern: 2 seq/GPU × 4 GPU = {args.workers} concurrent")
    print(f"Max tokens: {args.max_tokens}")
    print()

    t0 = time.monotonic()
    completed = 0

    # Stream results as they arrive
    async for result in fleet.stream():
        completed += 1
        status = "OK" if result.ok else f"ERR: {result.error[:60]}"
        print(
            f"  [{completed:>3}/{args.tasks}] "
            f"W{result.worker_id} {result.task_id}: "
            f"{result.completion_tokens:>5} tok, "
            f"{result.elapsed:>6.1f}s, "
            f"{result.tok_per_sec:>5.1f} tok/s — {status}"
        )

    wall = time.monotonic() - t0
    stats = fleet.stats

    print()
    print(f"=== Results ===")
    print(f"Total tasks:     {stats.total_completed + stats.total_errors}")
    print(f"Succeeded:       {stats.total_completed}")
    print(f"Errors:          {stats.total_errors}")
    print(f"Total tokens:    {stats.total_tokens:,}")
    print(f"Wall time:       {wall:.1f}s")
    print(f"Avg throughput:  {stats.total_tokens / wall:.1f} tok/s (aggregate)")
    if stats.total_completed > 0:
        avg_tok = stats.total_tokens / stats.total_completed
        print(f"Avg per task:    {avg_tok:.0f} tokens")
        print(f"Avg latency:     {wall / (stats.total_completed / args.workers):.1f}s per batch")
    print()
    print(f"Pipeline efficiency: {args.tasks} tasks in {wall:.0f}s")
    print(f"  Sequential estimate: {wall * args.workers / args.tasks * args.tasks:.0f}s")
    print(f"  Speedup: {args.workers:.0f}x (ideal) vs {wall / (wall / args.workers * args.tasks / args.tasks):.1f}x (actual)")


if __name__ == "__main__":
    asyncio.run(main())
