from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .state import manager, refresh_manager_settings
from .workbench import STATIC_DIR, router as workbench_router, start_sampler


@asynccontextmanager
async def lifespan(_: FastAPI):
    start_sampler()
    yield


app = FastAPI(title="Local LLM Control", version="0.1.0", lifespan=lifespan)
app.include_router(workbench_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse, include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/profiles")
def profiles() -> list[dict]:
    refresh_manager_settings()
    return manager.profiles()


@app.get("/v1/runtime")
def runtime_status() -> dict:
    return manager.status()


@app.post("/v1/runtime/{profile}/start")
async def start(profile: str) -> dict:
    try:
        refresh_manager_settings()
        return await manager.start(profile)
    except (KeyError, FileNotFoundError, MemoryError, RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/runtime/stop")
async def stop() -> dict:
    return await manager.stop()


@app.get("/v1/runtime/logs")
def logs(lines: int = Query(100, ge=1, le=2000)) -> dict[str, list[str]]:
    return {"lines": manager.tail(lines)}
