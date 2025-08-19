"""FastAPI app exposing trading endpoints with SSE streaming."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette import status as http_status
from starlette.responses import StreamingResponse

from .schemas import (
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    LoadDataRequest,
    LoadDataResponse,
    MetricsResponse,
    StatusResponse,
    SymbolsResponse,
    TrainRequest,
    TrainResponse,
)


RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

STATUS_INDEX: dict[str, dict[str, Any]] = {}
RUN_SUBSCRIBERS: dict[str, list[asyncio.Queue[str]]] = {}
SYMBOLS_CACHE = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

app = FastAPI(title="Trade Agent Service", version="0.1.0")

# CORS origins (configurable via ALLOWED_ORIGINS env, comma‑separated)
_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if _origins_env.strip():
    _origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:  # sensible local defaults
    _origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _publish(run_id: str, payload: dict[str, Any]) -> None:
    if run_id not in RUN_SUBSCRIBERS:
        return
    data = json.dumps(payload, default=str)
    for q in list(RUN_SUBSCRIBERS[run_id]):
        try:
            q.put_nowait(data)
        except Exception:  # pragma: no cover
            continue


def _persist(run_id: str) -> None:
    meta = STATUS_INDEX[run_id]
    path = RUNS_DIR / f"{run_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, default=str, indent=2)
    _publish(run_id, {"event": "persist", "status": meta.get("status")})


def _update(run_id: str, **updates: Any) -> None:
    meta = STATUS_INDEX[run_id]
    meta.update(updates)
    meta["updated_at"] = datetime.now(UTC)
    _persist(run_id)
    _publish(run_id, {"event": "update", **updates})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:  # pragma: no cover - trivial
    return HealthResponse(status="ok", version=app.version)


@app.get("/symbols", response_model=SymbolsResponse)
async def symbols() -> SymbolsResponse:
    return SymbolsResponse(symbols=SYMBOLS_CACHE)


def _run_data_pipeline(run_id: str, req: LoadDataRequest) -> None:
    try:
        from trade_agent.data.config import (
            CleaningConfig,
            DataPipelineConfig,
            DataSourceConfig,
            FeatureConfig,
            QualityConfig,
            StorageConfig,
            ValidationConfig,
        )
        from trade_agent.data.orchestrator import DataOrchestrator

        source = DataSourceConfig(
            name=req.source,
            type=req.source,
            symbols=req.symbols,
            start_date=str(req.start_date),
            end_date=str(req.end_date),
            interval=req.interval,
        )
        config = DataPipelineConfig(
            pipeline_name=req.pipeline_name,
            data_sources=[source],
            validation_config=ValidationConfig(),
            cleaning_config=CleaningConfig(),
            feature_config=FeatureConfig(),
            storage_config=StorageConfig(),
            quality_config=QualityConfig(),
            parallel_processing=False,
            n_workers=1,
        )
        _publish(run_id, {"event": "start", "stage": "data_pipeline"})
        orchestrator = DataOrchestrator(config)
        results = orchestrator.run_full_pipeline()
        ingestion = results.get("ingestion", {})
        total_records = sum(v.get("records", 0) for v in ingestion.values())
        _update(
            run_id,
            status="completed",
            message="Data load finished",
            records=total_records,
        )
    except Exception as e:  # pragma: no cover
        _update(run_id, status="failed", message=str(e))


@app.post("/data/load", response_model=LoadDataResponse)
async def load_data(
    req: LoadDataRequest, tasks: BackgroundTasks
) -> LoadDataResponse:
    run_id = str(uuid.uuid4())
    STATUS_INDEX[run_id] = {
        "run_id": run_id,
        "type": "data_load",
        "status": "running",
        "started_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "request": req.model_dump(),
    }
    _persist(run_id)
    tasks.add_task(_run_data_pipeline, run_id, req)
    return LoadDataResponse(run_id=run_id, status="running")


def _run_training(run_id: str, req: TrainRequest) -> None:
    try:
        import time
        for step in range(3):
            time.sleep(1)
            _update(run_id, message=f"training step {step+1}/3")
        metrics = {"reward_mean": 0.01, "drawdown": 0.05}
        _update(
            run_id,
            status="completed",
            message="Training complete",
            metrics=metrics,
        )
    except Exception as e:  # pragma: no cover
        _update(run_id, status="failed", message=str(e))


def require_token(env_var: str) -> Callable[[str], None]:
    """Return a dependency that validates an X-API-Token header.

    Each protected endpoint maps to an environment variable
    (e.g. TRAIN_ENDPOINT_TOKEN).
    If the env var is unset we return 500 to avoid silently disabling auth.
    """
    expected = os.getenv(env_var)

    def _verify(
        x_api_token: str = Header(..., alias="X-API-Token")
    ) -> None:  # type: ignore[override]
        if not expected:
            raise HTTPException(
                status_code=500,
                detail=f"Server token not configured: {env_var}",
            )
        if x_api_token != expected:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

    return _verify


@app.post("/train", response_model=TrainResponse)
async def train(
    req: TrainRequest,
    tasks: BackgroundTasks,
    _auth: None = Depends(require_token("TRAIN_ENDPOINT_TOKEN")),
) -> TrainResponse:
    run_id = str(uuid.uuid4())
    STATUS_INDEX[run_id] = {
        "run_id": run_id,
        "type": "train",
        "status": "running",
        "started_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "request": req.model_dump(),
        "linked_data_run": req.run_id,
    }
    _persist(run_id)
    tasks.add_task(_run_training, run_id, req)
    return TrainResponse(run_id=run_id, status="running")


def _run_backtest(run_id: str, req: BacktestRequest) -> None:
    try:
        import time
        time.sleep(1)
        metrics = {"sharpe": 1.2, "return": 0.15, "max_drawdown": 0.08}
        _update(
            run_id,
            status="completed",
            message="Backtest finished",
            metrics=metrics,
        )
    except Exception as e:  # pragma: no cover
        _update(run_id, status="failed", message=str(e))


@app.post("/backtest", response_model=BacktestResponse)
async def backtest(
    req: BacktestRequest,
    tasks: BackgroundTasks,
    _auth: None = Depends(require_token("BACKTEST_ENDPOINT_TOKEN")),
) -> BacktestResponse:
    run_id = str(uuid.uuid4())
    STATUS_INDEX[run_id] = {
        "run_id": run_id,
        "type": "backtest",
        "status": "running",
        "started_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "request": req.model_dump(),
    }
    _persist(run_id)
    tasks.add_task(_run_backtest, run_id, req)
    return BacktestResponse(run_id=run_id, status="running")


@app.get("/stream/runs/{run_id}")
async def stream_run(run_id: str):  # type: ignore[override]
    queue: asyncio.Queue[str] = asyncio.Queue()
    RUN_SUBSCRIBERS.setdefault(run_id, []).append(queue)

    async def event_generator():
        try:
            if run_id in STATUS_INDEX:
                snapshot = json.dumps(STATUS_INDEX[run_id], default=str)
                yield f"data: {snapshot}\n\n"
            while True:
                data = await queue.get()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:  # pragma: no cover
            pass
        finally:
            subs = RUN_SUBSCRIBERS.get(run_id, [])
            if queue in subs:
                subs.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/status/{run_id}", response_model=StatusResponse)
async def status(run_id: str) -> StatusResponse:
    meta = STATUS_INDEX.get(run_id)
    if not meta:
        path = RUNS_DIR / f"{run_id}.json"
        if path.exists():
            meta = json.loads(path.read_text())
            STATUS_INDEX[run_id] = meta
    if not meta:
        raise HTTPException(status_code=404, detail="run_id not found")
    return StatusResponse(
        run_id=run_id,
        type=meta["type"],
        status=meta["status"],
        started_at=datetime.fromisoformat(
            str(meta["started_at"]).replace("Z", "")
        ),
        updated_at=datetime.fromisoformat(
            str(meta["updated_at"]).replace("Z", "")
        ),
        message=meta.get("message"),
        meta={
            k: v
            for k, v in meta.items()
            if k
            not in {
                "run_id",
                "type",
                "status",
                "started_at",
                "updated_at",
                "message",
            }
        },
    )


@app.get("/metrics/{run_id}", response_model=MetricsResponse)
async def metrics(run_id: str) -> MetricsResponse:
    meta = STATUS_INDEX.get(run_id)
    if not meta:
        path = RUNS_DIR / f"{run_id}.json"
        if path.exists():
            meta = json.loads(path.read_text())
            STATUS_INDEX[run_id] = meta
    if not meta or "metrics" not in meta:
        raise HTTPException(
            status_code=404, detail="metrics not found for run_id"
        )
    return MetricsResponse(
        run_id=run_id,
        metrics=meta["metrics"],
        generated_at=datetime.now(UTC),
    )


__all__ = ["app"]
