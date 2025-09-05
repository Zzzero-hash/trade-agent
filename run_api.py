#!/usr/bin/env python3
"""
Run the FastAPI server for testing
"""
import uvicorn

from interface.api.main import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
