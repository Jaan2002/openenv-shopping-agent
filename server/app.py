from typing import Any

from fastapi import FastAPI, HTTPException

from env import ShoppingEnv
from inference import run
from models import Action, GradeRequest
from tasks import tasks

app = FastAPI()
env = ShoppingEnv()

# Automated evaluation expects scores strictly inside (0, 1), not exactly 0.0 / 1.0.
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


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    """OpenEnv runtime checks expect name + description as strings."""
    return {
        "name": "shopping-env",
        "description": "AI Shopping Assistant Environment",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    # OpenEnv runtime validation (openenv validate --url) expects status == "healthy"
    return {"status": "healthy", "ready": True, "tasks": len(tasks)}


@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    """Enumerate tasks + graders (required by Phase-2 style validators)."""
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
def grader(request: GradeRequest) -> dict[str, Any]:
    """Deterministic grading for a task + action; does not advance env task index."""
    try:
        score, details = env.grade(request.task_name, request.action)
        score = _strict_score(score)
        return {"score": score, "details": details}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: dict):
    act = Action(**action)
    obs, reward, done, info = env.step(act)
    return {
        "observation": obs.dict(),
        "reward": reward.score,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env.state()


@app.on_event("startup")
def run_inference_on_start():
    run()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
