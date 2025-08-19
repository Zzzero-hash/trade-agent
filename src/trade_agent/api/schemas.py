"""Pydantic schemas for service layer."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str | None = None


class SymbolsResponse(BaseModel):
    symbols: list[str]


class LoadDataRequest(BaseModel):
    symbols: list[str] = Field(..., description="List of symbol tickers")
    start_date: date
    end_date: date
    source: str = Field("yahoo_finance")
    interval: str = Field("1d")
    pipeline_name: str = Field("api_pipeline")


class LoadDataResponse(BaseModel):
    run_id: str
    status: str
    records: int | None = None


class TrainRequest(BaseModel):
    run_id: str | None = None  # optional link to data run
    algorithm: str = Field("ppo")
    config: dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    run_id: str
    status: str


class BacktestRequest(BaseModel):
    model_run_id: str
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = 10000.0


class BacktestResponse(BaseModel):
    run_id: str
    status: str


class StatusResponse(BaseModel):
    run_id: str
    type: str
    status: str
    started_at: datetime
    updated_at: datetime
    message: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    run_id: str
    metrics: dict[str, float]
    generated_at: datetime


__all__ = [
    "HealthResponse",
    "SymbolsResponse",
    "LoadDataRequest",
    "LoadDataResponse",
    "TrainRequest",
    "TrainResponse",
    "BacktestRequest",
    "BacktestResponse",
    "StatusResponse",
    "MetricsResponse",
]
