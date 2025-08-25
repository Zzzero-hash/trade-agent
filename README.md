# Local-First ML Trading Platform (Scaffold)

Phase 0 (M0) scaffold components:
- backend FastAPI service (`backend/main.py`)
- engine graph & example nodes (`engine/nodes.py`)
- shared feature flags (`shared/flags.py`)
- premium boundary (`premium/`)
- docs (`docs/ARCHITECTURE.md`)

Quick start:
```
make install
make test
make api  # starts FastAPI dev server
```

Create & execute example pipeline via HTTP:
```
POST /pipelines {"id":"demo","nodes":[
	{"id":"data","type":"data_source","symbol":"XYZ"},
	{"id":"fast","type":"sma","symbol":"XYZ","window":3},
	{"id":"slow","type":"sma","symbol":"XYZ","window":8},
	{"id":"x","type":"sma_crossover","symbol":"XYZ","fast":3,"slow":8}
]}
POST /pipelines/demo/execute
```
