"""FastAPI app: GET /health, POST /chat, GET /sessions/{session_id}/messages.

Every error leaves the server as JSON with a `detail` field — validation (422),
unknown session (404), all keys down (503) and even unexpected bugs (500).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, load_settings
from .db import Database
from .gateway import AllProvidersUnavailable, LLMGateway
from .schemas import ChatMetadata, ChatRequest, ChatResponse, HealthResponse
from .simulation import InvalidSimulationHeader, parse_simulation_header

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

ALL_PROVIDERS_UNAVAILABLE = "all providers unavailable"
SESSION_NOT_FOUND = "session not found"


def _clean_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """FastAPI-style error list, minus objects that are not JSON-serialisable (e.g. ctx.error)."""
    cleaned = []
    for err in errors:
        cleaned.append(
            {
                "type": err.get("type"),
                "loc": list(err.get("loc", ())),
                "msg": str(err.get("msg", "")).removeprefix("Value error, "),
            }
        )
    return cleaned


def create_app(
    settings: Optional[Settings] = None,
    gateway: Optional[LLMGateway] = None,
    database: Optional[Database] = None,
) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = database or Database(settings.database_url)  # schema is created here, on startup
        gw = gateway or LLMGateway(settings)
        app.state.db = db
        app.state.gateway = gw
        if len(gw.keys) < 2:
            logger.warning("LLM_API_KEYS has %d key(s); configure at least 2 for failover", len(gw.keys))
        logger.info(
            "ready: keys=%s cache=%s(%s) failure_simulation=%s db=%s",
            [k.key_id for k in gw.keys],
            gw.cache_enabled,
            settings.cache_type,
            settings.enable_failure_simulation,
            db.path,
        )
        yield
        if database is None:
            db.close()

    app = FastAPI(title="Resilient LLM Chatbot", version="1.0.0", lifespan=lifespan)

    origins = list(settings.allowed_origins) or ["http://localhost:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],  # includes Content-Type and X-Simulate-Failure
    )

    # ---- JSON-only errors -----------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _clean_validation_errors(exc.errors())})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    # ---- endpoints ------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> Any:
        db: Database = request.app.state.db
        gw: LLMGateway = request.app.state.gateway
        db_ok = db.ping()
        keys = len(gw.keys)
        body = {"status": "ok", "keys_configured": keys, "cache_enabled": gw.cache_enabled}
        if not db_ok or keys == 0:
            body["status"] = "unavailable"
            body["detail"] = "database unreachable" if not db_ok else "no API keys configured"
            return JSONResponse(status_code=503, content=body)
        return body

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        request: Request,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ) -> Any:
        started = time.perf_counter()
        db: Database = request.app.state.db
        gw: LLMGateway = request.app.state.gateway

        simulate: dict[str, int] = {}
        if settings.enable_failure_simulation and x_simulate_failure:
            try:
                simulate = parse_simulation_header(x_simulate_failure)
            except InvalidSimulationHeader as exc:
                return JSONResponse(status_code=422, content={"detail": f"invalid X-Simulate-Failure header: {exc}"})

        system_prompt = payload.system_prompt or db.get_system_prompt(payload.session_id) or settings.default_system_prompt
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m["role"], "content": m["content"]} for m in db.get_history(payload.session_id)]
        messages.append({"role": "user", "content": payload.message})

        try:
            result = await gw.complete(messages, simulate=simulate)
        except AllProvidersUnavailable as exc:
            logger.error("all %d keys failed for session %s: %s", len(gw.keys), payload.session_id, exc.attempts)
            return JSONResponse(status_code=503, content={"detail": ALL_PROVIDERS_UNAVAILABLE})

        metadata = ChatMetadata(
            resolution_source=result.resolution_source,
            model=result.model,
            latency_ms=int(round((time.perf_counter() - started) * 1000)),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )
        db.save_exchange(
            session_id=payload.session_id,
            user_message=payload.message,
            assistant_reply=result.reply,
            metadata=metadata.model_dump(),
            system_prompt=payload.system_prompt,
        )
        logger.info(
            "chat session=%s source=%s key=%s fallbacks=%d tokens=%d/%d latency=%dms",
            payload.session_id,
            metadata.resolution_source,
            metadata.key_id,
            metadata.fallback_attempts,
            metadata.input_tokens,
            metadata.output_tokens,
            metadata.latency_ms,
        )
        return ChatResponse(session_id=payload.session_id, reply=result.reply, metadata=metadata)

    @app.get("/sessions/{session_id}/messages")
    async def session_messages(session_id: str, request: Request) -> Any:
        db: Database = request.app.state.db
        if not db.session_exists(session_id):
            return JSONResponse(status_code=404, content={"detail": SESSION_NOT_FOUND})
        return {"session_id": session_id, "messages": db.get_history(session_id)}

    return app


app = create_app()
