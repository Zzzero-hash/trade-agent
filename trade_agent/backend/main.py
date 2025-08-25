from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Any, cast

from trade_agent.engine.nodes import (
    DataSourceNode,
    SmaTransformNode,
    SmaCrossoverSignalNode,
    execute_pipeline,
)
from shared import flags


class NodeSpec(BaseModel):
    id: str
    type: Literal["data_source", "sma", "sma_crossover"]
    symbol: str
    window: int | None = None
    fast: int | None = None
    slow: int | None = None


class PipelineSpec(BaseModel):
    id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    nodes: List[NodeSpec]


class PipelineOut(BaseModel):
    id: str
    node_count: int


_PIPELINES: Dict[str, PipelineSpec] = {}


def _instantiate(spec: PipelineSpec) -> List[Any]:
    instances: List[Any] = []
    for n in spec.nodes:
        if n.type == "data_source":
            instances.append(DataSourceNode(id=n.id, symbol=n.symbol))
        elif n.type == "sma":
            if n.window is None:
                raise HTTPException(
                    status_code=400,
                    detail="window required for sma node",
                )
            instances.append(
                SmaTransformNode(
                    id=n.id,
                    symbol=n.symbol,
                    window=n.window,
                )
            )
        elif n.type == "sma_crossover":
            if n.fast is None or n.slow is None:
                raise HTTPException(
                    status_code=400,
                    detail="fast and slow required for crossover node",
                )
            instances.append(
                SmaCrossoverSignalNode(
                    id=n.id,
                    symbol=n.symbol,
                    fast=n.fast,
                    slow=n.slow,
                )
            )
        else:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=400,
                detail=f"Unknown node type {n.type}",
            )
    return instances


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/flags")
async def list_flags() -> Dict[str, bool]:
    return flags.dump_all()


@router.post("/pipelines", response_model=PipelineOut, status_code=201)
async def create_pipeline(p: PipelineSpec) -> PipelineOut:
    if p.id in _PIPELINES:
        raise HTTPException(status_code=409, detail="Pipeline already exists")
    _PIPELINES[p.id] = p
    return PipelineOut(id=p.id, node_count=len(p.nodes))


@router.get("/pipelines/{pipeline_id}", response_model=PipelineSpec)
async def get_pipeline(pipeline_id: str) -> PipelineSpec:
    p = _PIPELINES.get(pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    return p


class ExecuteResponse(BaseModel):
    signals: List[Dict[str, Any]]
    count: int


@router.post(
    "/pipelines/{pipeline_id}/execute",
    response_model=ExecuteResponse,
)
async def execute(pipeline_id: str) -> ExecuteResponse:
    p = _PIPELINES.get(pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    nodes = _instantiate(p)
    ctx = execute_pipeline(nodes)
    signals_key = None
    # pick first signals key if present
    for k in ctx.keys():
        if k.startswith("signals:"):
            signals_key = k
            break
    raw: List[Dict[str, Any]] = []
    if signals_key:
        maybe = ctx.get(signals_key, [])
        if isinstance(maybe, list):
            # Best-effort cast; filter to dict entries only
            for item in maybe:
                if isinstance(item, dict):
                    raw.append(cast(Dict[str, Any], item))
    signals = raw
    return ExecuteResponse(signals=signals, count=len(signals))

app = FastAPI(title="Trade Platform API", version="0.0.1")
app.include_router(router)
