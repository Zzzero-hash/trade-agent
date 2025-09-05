from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interface.api.routers import data


app = FastAPI(title="Trade Agent API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router)


@app.get("/")
async def root():
    return {"message": "Trade Agent API"}
