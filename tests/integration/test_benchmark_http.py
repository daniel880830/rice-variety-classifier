import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rice_classifier.benchmark import benchmark_http
from rice_classifier.cli import app


class _PredictHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"label": 0, "class_name": "Cammeo", "probability": 0.9}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def server_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PredictHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_benchmark_http(server_url: str, grain: dict):
    result = benchmark_http(server_url, grain, n_requests=20)
    assert result["requests"] == 20
    assert result["p95_ms"] >= result["p50_ms"] > 0


def test_cli_benchmark_with_url(env: Path, server_url: str):
    runner = CliRunner()
    assert runner.invoke(app, ["train", "--model", "logistic_regression"]).exit_code == 0
    result = runner.invoke(app, ["benchmark", "--url", server_url])
    assert result.exit_code == 0, result.output
    assert "http" in json.loads((env / "reports" / "benchmark.json").read_text())
