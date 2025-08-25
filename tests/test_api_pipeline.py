import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from trade_agent.backend.main import app  # noqa: E402


def test_create_and_execute_pipeline():
    client = TestClient(app)
    payload = {
        "id": "demo",
        "nodes": [
            {"id": "data", "type": "data_source", "symbol": "XYZ"},
            {"id": "fast", "type": "sma", "symbol": "XYZ", "window": 3},
            {"id": "slow", "type": "sma", "symbol": "XYZ", "window": 8},
            {
                "id": "xover",
                "type": "sma_crossover",
                "symbol": "XYZ",
                "fast": 3,
                "slow": 8,
            },
        ],
    }
    r = client.post("/pipelines", json=payload)
    assert r.status_code == 201, r.text
    r2 = client.post("/pipelines/demo/execute")
    assert r2.status_code == 200
    data = r2.json()
    assert "signals" in data
    assert isinstance(data["signals"], list)
