"""Feature definitions and API request/response models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Column names exactly as they appear in the raw CSV.
RAW_FEATURES: tuple[str, ...] = (
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
)

# API field name -> raw column name.
API_TO_RAW: dict[str, str] = {
    "area": "Area",
    "perimeter": "Perimeter",
    "major_axis_length": "Major_Axis_Length",
    "minor_axis_length": "Minor_Axis_Length",
    "eccentricity": "Eccentricity",
    "convex_area": "Convex_Area",
    "extent": "Extent",
}

# Bounds are physical limits (positive sizes, ratios in [0, 1]) with generous headroom over the
# observed data, so plausible new grains are accepted but unit/typing mistakes are rejected.
# A value may be null: the trained pipeline imputes it exactly as it did during training.
PositiveSize = Annotated[float | None, Field(gt=0, le=100_000)]
Length = Annotated[float | None, Field(gt=0, le=2_000)]
Ratio = Annotated[float | None, Field(ge=0, le=1)]


class RiceGrain(BaseModel):
    """Morphological measurements of a single rice grain (pixels, from a grain image)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "area": 15231,
                "perimeter": 525.58,
                "major_axis_length": 229.75,
                "minor_axis_length": 85.09,
                "eccentricity": 0.9289,
                "convex_area": 15617,
                "extent": 0.5729,
            }
        },
    )

    area: PositiveSize = Field(description="Pixels inside the grain boundary.")
    perimeter: Length = Field(description="Boundary length in pixels.")
    major_axis_length: Length = Field(description="Longest axis of the fitted ellipse.")
    minor_axis_length: Length = Field(description="Shortest axis of the fitted ellipse.")
    eccentricity: Ratio = Field(description="Eccentricity of the fitted ellipse, in [0, 1].")
    convex_area: PositiveSize = Field(description="Pixels inside the convex hull.")
    extent: Ratio = Field(description="Grain area divided by its bounding-box area.")

    def to_raw(self) -> dict[str, float | None]:
        """Return the measurements keyed by raw training column names."""
        return {API_TO_RAW[name]: value for name, value in self.model_dump().items()}


class BatchRequest(BaseModel):
    """A batch of grains to classify."""

    model_config = ConfigDict(extra="forbid")

    items: list[RiceGrain] = Field(min_length=1)


class Prediction(BaseModel):
    """Prediction for one grain."""

    label: int = Field(description="Encoded class: 0 = Cammeo, 1 = Osmancik.")
    class_name: str = Field(description="Rice variety.")
    probability: float = Field(description="Predicted probability of the returned class.")


class BatchResponse(BaseModel):
    """Predictions for a batch, in request order."""

    predictions: list[Prediction]


class HealthResponse(BaseModel):
    """Service liveness and readiness."""

    status: str
    model_loaded: bool
