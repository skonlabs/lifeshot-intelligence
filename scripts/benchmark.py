#!/usr/bin/env python
"""Tiny latency benchmark for the face-detect and moderation paths.

Sends N requests and prints p50/p95. Point it at a running instance:

    python scripts/benchmark.py --url http://127.0.0.1:8000 \
        --key dev-local-key-please-change --image sample.jpg --n 30
"""
from __future__ import annotations

import argparse
import statistics
import time

import httpx


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = int(round((p / 100) * (len(values) - 1)))
    return values[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--endpoint", default="/v1/intelligence/face/detect")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        img = f.read()

    headers = {"X-API-Key": args.key}
    timings = []
    with httpx.Client(base_url=args.url, timeout=60.0) as client:
        for i in range(args.n):
            t0 = time.perf_counter()
            resp = client.post(args.endpoint, headers=headers, files={"file": ("img.jpg", img)})
            dt = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            timings.append(dt)
            print(f"  req {i+1}/{args.n}: {dt:.1f} ms")

    print("\n--- results ---")
    print(f"count : {len(timings)}")
    print(f"p50   : {statistics.median(timings):.1f} ms")
    print(f"p95   : {percentile(timings, 95):.1f} ms")
    print(f"max   : {max(timings):.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
