"""Command-line interface: ``rice train | leakage-audit | benchmark | serve``."""

from __future__ import annotations

import json
import logging
from typing import Annotated

import typer

from rice_classifier.config import get_settings, load_params
from rice_classifier.data import Dataset, load_dataset
from rice_classifier.schema import RAW_FEATURES, RiceGrain
from rice_classifier.train import run_experiment, run_leakage_audit, save_json, save_model

app = typer.Typer(help="Rice variety classifier.", no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Rice variety classifier."""
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load() -> tuple[Dataset, dict]:
    settings = get_settings()
    params = load_params(settings.resolve(settings.params_path))
    dataset = load_dataset(
        settings.resolve(settings.data_path),
        target=params["data"]["target"],
        label_map=params["data"]["label_map"],
        expected_features=RAW_FEATURES,
    )
    return dataset, params


@app.command()
def train(
    model: Annotated[
        list[str] | None, typer.Option(help="Only these [models.*] entries (repeatable).")
    ] = None,
    save: Annotated[bool, typer.Option(help="Save the selected model artifact.")] = True,
) -> None:
    """Grid-search every candidate, report test metrics and save the CV-selected model."""
    settings = get_settings()
    dataset, params = _load()
    result = run_experiment(dataset, params, models=model)
    path = save_json(result.report, settings.resolve(settings.reports_dir) / "metrics.json")
    typer.echo(f"Report written to {path}")
    for name, res in result.report["models"].items():
        cv, test = res["cv_accuracy"], res["test"]
        ci = test["accuracy_ci95"]
        typer.echo(
            f"  {name:<24} CV {cv['mean']:.4f} ± {cv['std']:.4f}  test {test['accuracy']:.4f} "
            f"[{ci['low']:.4f}, {ci['high']:.4f}]  AUC {test['roc_auc']:.4f}  {res['best_params']}"
        )
    typer.echo(f"Selected by CV: {result.report['selected_model']}")
    if save:
        path = save_model(result.model, result.metadata, settings.resolve(settings.model_path))
        typer.echo(f"Model saved to {path}")


@app.command("leakage-audit")
def leakage_audit() -> None:
    """Compare CV accuracy with preprocessing fit inside vs. outside the folds."""
    settings = get_settings()
    reports_dir = settings.resolve(settings.reports_dir)
    metrics = reports_dir / "metrics.json"
    if not metrics.is_file():
        typer.echo("Run `rice train` first: reports/metrics.json not found.", err=True)
        raise typer.Exit(code=2)
    dataset, params = _load()
    audit = run_leakage_audit(dataset, json.loads(metrics.read_text("utf-8")), params)
    save_json(audit, reports_dir / "leakage_audit.json")
    for name, row in audit.items():
        typer.echo(
            f"  {name:<24} leak-free {row['leak_free']:.4f}  leaky {row['leaky']:.4f}  "
            f"gap {row['gap_pp']:+.2f} pp"
        )


@app.command()
def benchmark(
    url: Annotated[str | None, typer.Option(help="Also time a running API at this URL.")] = None,
) -> None:
    """Measure inference latency/throughput and write reports/benchmark.json."""
    from rice_classifier.benchmark import (
        artifact_size_kb,
        benchmark_http,
        benchmark_model,
        environment,
    )
    from rice_classifier.predict import load_model

    settings = get_settings()
    model_path = settings.resolve(settings.model_path)
    bundle = load_model(model_path)
    dataset, _ = _load()
    result = {
        "model": bundle.metadata["model_name"],
        "artifact_kb": artifact_size_kb(model_path),
        "in_process": benchmark_model(bundle, dataset.X),
        "environment": environment(),
    }
    if url:
        example = RiceGrain.model_config["json_schema_extra"]["example"]  # type: ignore[index]
        result["http"] = benchmark_http(url, example)
    save_json(result, settings.resolve(settings.reports_dir) / "benchmark.json")
    typer.echo(json.dumps(result, indent=2))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Defaults to RICE_API_HOST.")] = None,
    port: Annotated[int | None, typer.Option(help="Defaults to RICE_API_PORT.")] = None,
) -> None:
    """Start the inference API."""
    import uvicorn

    from rice_classifier.api import create_app

    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=host or settings.api_host,
        port=port or settings.api_port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
