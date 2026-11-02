"""Latency and throughput measurements for the saved model and the HTTP API."""

from __future__ import annotations

import json
import platform
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from rice_classifier.predict import ModelBundle, predict_records


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = np.asarray(samples_ms)
    return {
        "p50_ms": round(float(np.percentile(ordered, 50)), 3),
        "p95_ms": round(float(np.percentile(ordered, 95)), 3),
        "mean_ms": round(statistics.fmean(samples_ms), 3),
    }


def benchmark_model(
    bundle: ModelBundle, X: pd.DataFrame, *, n_single: int = 500, batch_size: int = 1000
) -> dict[str, Any]:
    """Time in-process single-row predictions and batch throughput.

    Args:
        bundle: Loaded model.
        X: Raw feature rows to sample requests from.
        n_single: Number of timed single-row predictions.
        batch_size: Rows per timed batch.

    Returns:
        Single-row latency percentiles and batch throughput in rows/second.
    """
    records = X.astype(object).where(X.notna(), None).to_dict(orient="records")
    predict_records(bundle, records[:5])  # warm-up

    single = []
    for i in range(n_single):
        start = time.perf_counter()
        predict_records(bundle, [records[i % len(records)]])
        single.append((time.perf_counter() - start) * 1000)

    batch = [records[i % len(records)] for i in range(batch_size)]
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        predict_records(bundle, batch)
        runs.append(time.perf_counter() - start)
    best = min(runs)
    return {
        "single_row": _percentiles(single),
        "batch": {
            "rows": batch_size,
            "seconds": round(best, 4),
            "rows_per_second": round(batch_size / best),
        },
    }


def benchmark_http(url: str, payload: dict[str, Any], *, n_requests: int = 300) -> dict[str, Any]:
    """Time sequential ``POST /predict`` calls against a running server.

    Args:
        url: Base URL such as ``http://127.0.0.1:8000``.
        payload: A valid single-grain request body.
        n_requests: Number of timed requests.

    Returns:
        Latency percentiles and sequential requests/second.
    """
    body = json.dumps(payload).encode()

    def call() -> None:
        request = urllib.request.Request(
            f"{url.rstrip('/')}/predict", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()

    for _ in range(10):
        call()
    samples = []
    start_all = time.perf_counter()
    for _ in range(n_requests):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    total = time.perf_counter() - start_all
    return {
        **_percentiles(samples),
        "requests": n_requests,
        "requests_per_second": round(n_requests / total),
    }


def environment() -> dict[str, str]:
    """Hardware and library versions the numbers were measured on."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }


def artifact_size_kb(path: Path) -> float:
    """Size of the model artifact in KiB."""
    return round(path.stat().st_size / 1024, 1)
