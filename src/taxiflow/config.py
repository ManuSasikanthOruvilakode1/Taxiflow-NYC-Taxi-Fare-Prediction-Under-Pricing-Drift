"""Configuration models for Taxiflow using Pydantic."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

MonthRange = tuple[str, str]


class FilterConfig(BaseModel):
    min_fare: float = 3.00
    max_fare: float = 500.00
    min_distance_km: float = 0.20
    max_distance_km: float = 120.00
    min_duration_min: float = 1.00
    max_duration_min: float = 180.00
    min_speed_kmph: float = 1.00
    max_speed_kmph: float = 115.00


class BadRecordRule(BaseModel):
    month: str
    vendor_id: int
    payment_type: int
    reason: str = ""


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    start_month: str = "2024-01"
    end_month: str = "2026-08"
    known_bad_records: list[BadRecordRule] = Field(default_factory=list)
    filters: FilterConfig = Field(default_factory=FilterConfig)


class SplitConfig(BaseModel):
    train: MonthRange = ("2025-12", "2026-04")
    validation: MonthRange = ("2026-05", "2026-05")
    calibration: MonthRange = ("2026-06", "2026-06")
    test: MonthRange = ("2026-07", "2026-08")
    sample_per_month: int = 1_000_000


class DriftModelConfig(BaseModel):
    name: str
    train: MonthRange
    validation: MonthRange
    sample_per_month: int = 300_000


class SeasonalityConfig(BaseModel):
    enabled: bool = True
    fit_window: MonthRange = ("2024-01", "2025-11")


class WalkForwardConfig(BaseModel):
    test_months: list[str] = Field(default_factory=lambda: ["2026-07", "2026-08"])
    train_months: int = 4
    sample_per_month: int = 1_000_000


class ModelConfig(BaseModel):
    type: str = "lightgbm"
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50
    params: dict[str, Any] = Field(default_factory=dict)


class ConformalConfig(BaseModel):
    confidence_level: float = 0.90


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "models/taxiflow_lgbm_serving.joblib"


class AppConfig(BaseModel):
    project_name: str = "taxiflow"
    version: str = "1.0.0"
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    drift_models: list[DriftModelConfig] = Field(default_factory=list)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        project = raw.get("project", {})
        return cls(
            project_name=project.get("name", "taxiflow"),
            version=project.get("version", "1.0.0"),
            seed=project.get("seed", 42),
            data=DataConfig(**raw.get("data", {})),
            split=SplitConfig(**raw.get("split", {})),
            drift_models=[DriftModelConfig(**d) for d in raw.get("drift_models", [])],
            walk_forward=WalkForwardConfig(**raw.get("walk_forward", {})),
            seasonality=SeasonalityConfig(**raw.get("seasonality", {})),
            model=ModelConfig(**raw.get("model", {})),
            conformal=ConformalConfig(**raw.get("conformal", {})),
            serving=ServingConfig(**raw.get("serving", {})),
        )
