"""Fleet — 8-worker task queue for pipelined inference.

隙間なく次の仕事を片付ける8人の職人。
vLLM への同時リクエスト数を 8 に固定し (2 seq/GPU × 4 GPU)、
タスクキューから順次取り出して処理する。

Usage:
    fleet = Fleet("http://localhost:8014/v1/chat/completions", workers=8)

    # Submit tasks
    fleet.submit("Read this paper and summarize", task_id="paper_001")
    fleet.submit("Analyze this codebase", task_id="code_002")
    ...

    # Run all queued tasks
    results = await fleet.run()

    # Or use as async generator for streaming results
    async for result in fleet.stream():
        print(f"[{result.task_id}] done: {result.completion_tokens} tokens")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A single inference task."""
    task_id: str
    messages: list[dict]
    max_tokens: int = 131072
    temperature: float = 0.6
    system_prompt: str | None = None


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    worker_id: int
    content: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    elapsed: float
    tok_per_sec: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class FleetStats:
    """Running statistics for the fleet."""
    total_submitted: int = 0
    total_completed: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    start_time: float = 0.0
    worker_busy: list[bool] = field(default_factory=lambda: [False] * 8)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time if self.start_time else 0.0

    @property
    def throughput(self) -> float:
        return self.total_tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def active_workers(self) -> int:
        return sum(self.worker_busy)


class Fleet:
    """8-worker pipelined inference queue.

    Maintains exactly `workers` concurrent requests to vLLM.
    When a worker finishes, it immediately picks the next task from the queue.
    Zero idle time between tasks.
    """

    def __init__(
        self,
        url: str = "http://localhost:8014/v1/chat/completions",
        model: str | None = None,
        workers: int = 8,
        timeout: int = 3600,
    ):
        self.url = url
        self.model = model
        self.workers = workers
        self.timeout = timeout
        self._queue: asyncio.Queue[Task | None] = asyncio.Queue()
        self._results: asyncio.Queue[TaskResult] = asyncio.Queue()
        self.stats = FleetStats(worker_busy=[False] * workers)

    def submit(
        self,
        prompt: str,
        task_id: str | None = None,
        max_tokens: int = 131072,
        temperature: float = 0.6,
        system_prompt: str | None = None,
    ) -> str:
        """Submit a task to the queue. Returns task_id."""
        tid = task_id or f"task_{self.stats.total_submitted:04d}"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        task = Task(
            task_id=tid,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        self._queue.put_nowait(task)
        self.stats.total_submitted += 1
        logger.debug("Submitted %s (queue size: %d)", tid, self._queue.qsize())
        return tid

    def submit_messages(
        self,
        messages: list[dict],
        task_id: str | None = None,
        max_tokens: int = 131072,
        temperature: float = 0.6,
    ) -> str:
        """Submit a task with custom messages list."""
        tid = task_id or f"task_{self.stats.total_submitted:04d}"
        task = Task(task_id=tid, messages=messages,
                    max_tokens=max_tokens, temperature=temperature)
        self._queue.put_nowait(task)
        self.stats.total_submitted += 1
        return tid

    async def _discover_model(self, session: aiohttp.ClientSession) -> str:
        """Auto-discover model name from vLLM."""
        if self.model:
            return self.model
        api_base = self.url.split("/v1/")[0]
        async with session.get(f"{api_base}/v1/models") as resp:
            data = await resp.json()
            self.model = data["data"][0]["id"]
        return self.model

    async def _worker(
        self,
        worker_id: int,
        session: aiohttp.ClientSession,
        model: str,
    ) -> None:
        """Single worker loop: pull task, execute, repeat until sentinel."""
        while True:
            task = await self._queue.get()
            if task is None:  # Sentinel: shutdown
                self._queue.task_done()
                break

            self.stats.worker_busy[worker_id] = True
            t0 = time.monotonic()

            try:
                payload = {
                    "model": model,
                    "messages": task.messages,
                    "max_tokens": task.max_tokens,
                    "temperature": task.temperature,
                }

                async with session.post(self.url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        result = TaskResult(
                            task_id=task.task_id,
                            worker_id=worker_id,
                            content="",
                            reasoning=None,
                            prompt_tokens=0,
                            completion_tokens=0,
                            elapsed=time.monotonic() - t0,
                            tok_per_sec=0.0,
                            error=f"HTTP {resp.status}: {text[:500]}",
                        )
                    else:
                        data = await resp.json()
                        elapsed = time.monotonic() - t0
                        usage = data.get("usage", {})
                        choice = data["choices"][0]["message"]
                        comp_tokens = usage.get("completion_tokens", 0)

                        result = TaskResult(
                            task_id=task.task_id,
                            worker_id=worker_id,
                            content=choice.get("content", ""),
                            reasoning=choice.get("reasoning_content"),
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=comp_tokens,
                            elapsed=round(elapsed, 2),
                            tok_per_sec=round(comp_tokens / elapsed, 1) if elapsed > 0 else 0.0,
                        )

            except Exception as e:
                result = TaskResult(
                    task_id=task.task_id,
                    worker_id=worker_id,
                    content="",
                    reasoning=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    elapsed=time.monotonic() - t0,
                    tok_per_sec=0.0,
                    error=str(e)[:500],
                )

            self.stats.worker_busy[worker_id] = False

            if result.ok:
                self.stats.total_completed += 1
                self.stats.total_tokens += result.completion_tokens
            else:
                self.stats.total_errors += 1

            await self._results.put(result)
            self._queue.task_done()

            logger.info(
                "Worker %d finished %s: %d tokens in %.1fs (%.1f tok/s)%s",
                worker_id, task.task_id, result.completion_tokens,
                result.elapsed, result.tok_per_sec,
                f" ERROR: {result.error}" if result.error else "",
            )

    async def run(self) -> list[TaskResult]:
        """Run all queued tasks, return results when all complete."""
        self.stats.start_time = time.monotonic()
        total_tasks = self._queue.qsize()

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            model = await self._discover_model(session)

            logger.info(
                "Fleet launching: %d tasks, %d workers, model=%s",
                total_tasks, self.workers, model.split("/")[-1],
            )

            # Start workers
            workers = [
                asyncio.create_task(self._worker(i, session, model))
                for i in range(self.workers)
            ]

            # Wait for all tasks to be processed
            await self._queue.join()

            # Send shutdown sentinels
            for _ in range(self.workers):
                await self._queue.put(None)
            await asyncio.gather(*workers)

        # Collect results
        results = []
        while not self._results.empty():
            results.append(self._results.get_nowait())

        elapsed = time.monotonic() - self.stats.start_time
        logger.info(
            "Fleet complete: %d/%d ok, %d tokens, %.1fs, %.1f tok/s avg",
            self.stats.total_completed, total_tasks,
            self.stats.total_tokens, elapsed, self.stats.throughput,
        )

        return results

    async def stream(self) -> AsyncIterator[TaskResult]:
        """Run tasks and yield results as they complete."""
        self.stats.start_time = time.monotonic()
        total_tasks = self._queue.qsize()
        received = 0

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            model = await self._discover_model(session)

            workers = [
                asyncio.create_task(self._worker(i, session, model))
                for i in range(self.workers)
            ]

            while received < total_tasks:
                result = await self._results.get()
                received += 1
                yield result

            # Shutdown
            for _ in range(self.workers):
                await self._queue.put(None)
            await asyncio.gather(*workers)
