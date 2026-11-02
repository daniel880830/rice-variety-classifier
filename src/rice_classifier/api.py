"""FastAPI inference service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from rice_classifier import __version__
from rice_classifier.config import Settings, get_settings
from rice_classifier.predict import (
    ModelBundle,
    ModelLoadError,
    ModelNotFoundError,
    load_model,
    predict_records,
)
from rice_classifier.schema import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    Prediction,
    RiceGrain,
)

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    loader: Callable[[Path], ModelBundle] = load_model,
) -> FastAPI:
    """Build the application.

    The model is loaded once at startup. If loading fails the service still starts
    (so ``/health`` can report it) and prediction endpoints return 503.

    Args:
        settings: Runtime settings; defaults to environment-derived settings.
        loader: Function that loads the model artifact (injectable for tests).

    Returns:
        The configured FastAPI app.
    """
    settings = settings or get_settings()
    model_path = settings.resolve(settings.model_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.bundle = None
        try:
            app.state.bundle = loader(model_path)
            logger.info(
                "Loaded model %s from %s", app.state.bundle.metadata["model_name"], model_path
            )
        except (ModelNotFoundError, ModelLoadError) as exc:
            logger.error("Model not loaded: %s", exc)
        yield

    app = FastAPI(
        title="Rice Variety Classifier",
        version=__version__,
        description="Classifies a rice grain as Cammeo or Osmancik from seven shape measurements.",
        lifespan=lifespan,
    )

    def get_bundle(request: Request) -> ModelBundle:
        bundle = getattr(request.app.state, "bundle", None)
        if bundle is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not loaded")
        return bundle

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health(request: Request) -> HealthResponse:
        """Liveness, plus whether the model is ready to serve."""
        loaded = getattr(request.app.state, "bundle", None) is not None
        return HealthResponse(status="ok", model_loaded=loaded)

    @app.get("/model/info")
    def model_info(request: Request) -> dict[str, Any]:
        """Training metadata of the loaded model."""
        return get_bundle(request).metadata

    @app.post("/predict")
    def predict(grain: RiceGrain, request: Request) -> Prediction:
        """Classify a single grain."""
        return predict_records(get_bundle(request), [grain.to_raw()])[0]

    @app.post("/predict/batch")
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        """Classify up to ``RICE_MAX_BATCH_SIZE`` grains in one call."""
        if len(batch.items) > settings.max_batch_size:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"Batch of {len(batch.items)} exceeds the limit of {settings.max_batch_size}",
            )
        bundle = get_bundle(request)
        return BatchResponse(predictions=predict_records(bundle, [g.to_raw() for g in batch.items]))

    return app
