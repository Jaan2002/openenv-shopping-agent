import logging
import os
import threading
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import (
    EnvironmentMetadata,
    HealthResponse,
    HealthStatus,
    ResetRequest,
    ResetResponse,
    State,
    StepRequest,
    StepResponse,
)

from env import ShoppingEnv
from models import GradeRequest, ShoppingAction, ShoppingObservation
from tasks import tasks

app = FastAPI(
    title="OpenEnv Shopping Environment",
    version="1.0.0",
    description="Multi-criteria shopping tasks with deterministic graders.",
)

# Single process-local env — required for sequential task_index (OpenEnv HTTP
# create_app() instantiates a new Environment per request and would break state).
env = ShoppingEnv()

SCORE_FLOOR = 0.001
SCORE_CEILING = 0.999


def _strict_score(value: float) -> float:
    score = round(float(value), 4)
    if score <= SCORE_FLOOR:
        return SCORE_FLOOR
    if score >= SCORE_CEILING:
        return SCORE_CEILING
    return score


@app.get("/")
def home():
    return {"message": "Shopping AI Env Running"}


@app.get("/metadata", response_model=EnvironmentMetadata)
def metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="shopping-env",
        description="AI Shopping Assistant Environment",
        version="0.1.0",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status=HealthStatus.HEALTHY)


@app.get("/schema")
def schema() -> dict[str, Any]:
    """OpenEnv runtime validation expects action, observation, and state JSON schemas."""
    return {
        "action": ShoppingAction.model_json_schema(),
        "observation": ShoppingObservation.model_json_schema(),
        "state": State.model_json_schema(),
    }


@app.post("/mcp")
def mcp(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Minimal JSON-RPC envelope for openenv validate --url (POST /mcp)."""
    body = payload or {}
    return {"jsonrpc": "2.0", "id": body.get("id"), "result": {}}


@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tasks:
        g = t.get("grader") or {}
        out.append(
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "category": t.get("category"),
                "difficulty": t.get("difficulty", t.get("name")),
                "grader": g,
            }
        )
    return out


@app.post("/grader")
def grader_endpoint(request: GradeRequest) -> dict[str, Any]:
    try:
        score, details = env.grade(request.task_name, request.action)
        score = _strict_score(score)
        return {"score": score, "details": details}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    _ = request  # seed / episode_id accepted for API compatibility
    obs = env.reset()
    p = serialize_observation(obs)
    return ResetResponse(
        observation=p["observation"],
        reward=p["reward"],
        done=p["done"],
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    try:
        act = ShoppingAction.model_validate(request.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    obs = env.step(act)
    p = serialize_observation(obs)
    return StepResponse(
        observation=p["observation"],
        reward=p["reward"],
        done=p["done"],
    )


@app.get("/state")
def state_endpoint() -> State:
    d = env.state()
    return State(
        episode_id=str(d.get("current_task") or ""),
        step_count=int(d.get("episode_count", 0)),
        status=d.get("status", "running"),
        total_tasks=d.get("total_tasks"),
    )


def _inference_thread_target() -> None:
    """Runs shopping inference after the server is accepting traffic (logs for eval harness)."""
    try:
        from inference import run

        run()
    except Exception:
        logging.exception("Background inference failed")


@app.on_event("startup")
def run_inference_on_start() -> None:
    """
    Do not block startup: Hugging Face Spaces wait for the process to listen on PORT.
    Synchronous LLM inference would keep the app in 'Building/Loading' forever.
    """
    if os.getenv("RUN_INFERENCE_ON_STARTUP", "true").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    t = threading.Thread(
        target=_inference_thread_target,
        name="openenv-inference",
        daemon=True,
    )
    t.start()


def main():
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
