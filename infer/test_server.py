#!/usr/bin/env python3

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    print("🚀 Test server starting...")
    print("✅ Test server startup complete!")
    
    yield  # Server is running
    
    # Shutdown
    print("🛑 Test server shutting down...")

app = FastAPI(
    title="Test API",
    description="Simple test API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Test API is running!"}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind to")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "test_server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=False
    )