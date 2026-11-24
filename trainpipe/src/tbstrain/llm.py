"""Bounded async Anthropic Messages client with validated, atomic response caching."""

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .core import atomic, digest, dumps, loads, read


def post(url, headers, body, timeout):
    request = Request(url, data=dumps(body).encode(), headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return loads(response.read())


class Client:
    def __init__(self, cfg, cache, transport=None):
        self.cfg = cfg
        self.cache = cache
        self.transport = transport
        self.workers = int(cfg.get("concurrency", 64))
        if self.workers < 1 or cfg.get("retries", 4) < 0:
            raise ValueError("invalid concurrency or retries")
        self.executor = None
        self.inflight = {}
        self.stats = Counter()

    async def request(self, rubric, prompt, parse):
        key = digest([self.cfg["base_url"], self.cfg["model"], rubric, prompt])
        path = self.cache / (key + ".json")
        cached = read(path)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return parse(cached["answer"], prompt)
        if key in self.inflight:
            return await self.inflight[key]
        task = asyncio.create_task(self.fetch(path, rubric, prompt, parse))
        self.inflight[key] = task
        try:
            return await task
        finally:
            self.inflight.pop(key, None)

    async def fetch(self, path, rubric, prompt, parse):
        cfg = self.cfg
        url = cfg["base_url"].rstrip("/")
        url += "/messages" if url.endswith("/v1") else "/v1/messages"
        body = dict(
            model=cfg["model"],
            system=rubric,
            max_tokens=cfg.get("max_tokens", 3000),
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if self.transport is None:
            headers["x-api-key"] = os.environ[cfg["api_key_env"]]
        for attempt in range(cfg.get("retries", 4) + 1):
            try:
                self.stats["requests"] += 1
                args = (url, headers, body, cfg.get("timeout", 180))
                if self.transport:
                    response = await self.transport(*args)
                else:
                    response = await asyncio.get_running_loop().run_in_executor(
                        self.executor, partial(post, *args)
                    )
                break
            except HTTPError as exc:
                if (exc.code != 429 and not 500 <= exc.code < 600) or attempt >= cfg.get("retries", 4):
                    raise
                await asyncio.sleep(min(60, cfg.get("backoff", 1) * 2**attempt))
        usage = dict(response.get("usage", {}))
        self.account(usage)
        raw = self.text(response)
        try:
            answer = self.decode(raw)
            value = parse(answer, prompt)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            # one repair turn: show the model its answer and the validation error
            self.stats["repairs"] += 1
            body["messages"] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": raw or "(empty)"},
                {"role": "user", "content": f"That answer was rejected: {exc}. Quotes must be copied exactly from "
                 "the material. Return the corrected JSON object only."},
            ]
            args = (url, headers, body, cfg.get("timeout", 180))
            response = await (self.transport(*args) if self.transport else
                              asyncio.get_running_loop().run_in_executor(self.executor, partial(post, *args)))
            extra = response.get("usage", {})
            self.account(extra)
            for k, v in extra.items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
            raw2 = self.text(response)
            try:
                answer = self.decode(raw2)
                value = parse(answer, prompt)
            except Exception as exc2:
                # keep the rejected answers for auditing the rubric / validator (not a cache entry)
                reject = path.parent.parent / ".judge-rejects" / path.name
                atomic(reject, {"first": raw, "first_error": str(exc), "second": raw2, "second_error": str(exc2),
                                "prompt_sha": digest(prompt)})
                raise
        atomic(path, {"answer": answer, "usage": usage})
        return value

    def account(self, usage):
        cfg = self.cfg
        for k in ("input_tokens", "output_tokens"):
            n = usage.get(k, 0) or 0
            self.stats[k] += n
            self.stats["cost_usd"] += n * cfg.get(k + "_per_million", 0) / 1e6

    @staticmethod
    def text(response):
        return "".join(b["text"] for b in response.get("content", []) if b.get("type") == "text").strip()

    @staticmethod
    def decode(raw):
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no JSON object in the answer")
        return loads(raw[start : end + 1])

    async def map(self, jobs, process):
        iterator = iter(jobs)

        async def worker():
            for job in iterator:
                await process(job)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            self.executor = pool
            async with asyncio.TaskGroup() as group:
                for _ in range(self.workers):
                    group.create_task(worker())
