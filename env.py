"""
ShoppingEnv — OpenEnv-style shopping assistant (singleton-friendly).

reset() → step(action) advances the rotating task list. Rewards live on
ShoppingObservation.reward in (0, 1), exclusive of 0.0 and 1.0 after clamping.
"""

from __future__ import annotations

from typing import Any, ClassVar, List

from models import ShoppingAction, ShoppingObservation, Product
from tasks import tasks


class ShoppingEnv:
    """Stateful environment; use one shared instance behind the FastAPI server."""

    TASKS: ClassVar[List[Any]] = tasks

    def __init__(self) -> None:
        self.current_task_index: int = 0
        self.current_task: dict = tasks[0]
        self._episode_count: int = 0

    def reset(self) -> ShoppingObservation:
        if self.current_task_index >= len(tasks):
            self.current_task_index = 0

        self.current_task = tasks[self.current_task_index]
        self._episode_count += 1

        return self._build_observation(self.current_task, done=False, reward=None)

    @staticmethod
    def compute_reward(task: dict, action: ShoppingAction) -> tuple[float, dict]:
        products = task["products"]
        grader = task.get("grader", {})

        selected = None
        for p in products:
            if p["name"].strip().lower() == action.action_type.strip().lower():
                selected = p
                break

        breakdown: dict = {}

        if selected is None:
            score = 0.15
            breakdown["reason"] = "product_not_found"
        else:
            optimal_name = grader.get("target", task.get("optimal", ""))
            is_optimal = selected["name"] == optimal_name

            if is_optimal:
                score = grader.get("score_if_correct")
                if score is None:
                    score = grader.get("correct_score", 0.75)
                breakdown["reason"] = "optimal_match"
            else:
                score = grader.get("score_if_wrong")
                if score is None:
                    score = grader.get("incorrect_score", 0.30)
                breakdown["reason"] = "suboptimal"

            breakdown["base"] = score

            bonus = 0.0

            if selected["price"] <= task["budget"]:
                bonus += 0.05
                breakdown["budget_bonus"] = 0.05
            else:
                bonus -= 0.08
                breakdown["budget_penalty"] = -0.08

            priority = task.get("priority", "")
            if priority == "price":
                best = min(products, key=lambda x: x["price"])
                if selected["name"] == best["name"]:
                    bonus += 0.05
                    breakdown["priority_bonus"] = 0.05
            elif priority == "rating":
                best = max(products, key=lambda x: x["rating"])
                if selected["name"] == best["name"]:
                    bonus += 0.05
                    breakdown["priority_bonus"] = 0.05
            elif priority == "battery":
                best = max(products, key=lambda x: x["battery"])
                if selected["name"] == best["name"]:
                    bonus += 0.05
                    breakdown["priority_bonus"] = 0.05

            score += bonus

        score = float(round(min(0.99, max(0.01, score)), 4))

        info = {
            "optimal": task.get("optimal"),
            "selected": action.action_type,
            "score_breakdown": breakdown,
            "task_name": task.get("name"),
        }
        return score, info

    def grade(self, task_name: str, action: ShoppingAction) -> tuple[float, dict]:
        for t in tasks:
            if t.get("name") == task_name:
                return self.compute_reward(t, action)
        raise ValueError(f"Unknown task_name: {task_name}")

    def step(self, action: ShoppingAction) -> ShoppingObservation:
        task = self.current_task
        score, info = self.compute_reward(task, action)
        self.current_task_index += 1

        obs = self._build_observation(task, done=True, reward=score)
        meta = {**obs.metadata, "info": info}
        return obs.model_copy(update={"metadata": meta})

    def state(self) -> dict:
        return {
            "status": "running",
            "episode_count": self._episode_count,
            "current_task": self.current_task.get("name"),
            "total_tasks": len(tasks),
        }

    @staticmethod
    def _build_observation(
        task: dict,
        *,
        done: bool,
        reward: float | None,
    ) -> ShoppingObservation:
        return ShoppingObservation(
            done=done,
            reward=reward,
            metadata={},
            user_need=task["user_need"],
            budget=task["budget"],
            category=task["category"],
            priority=task["priority"],
            products=[Product(**p) for p in task["products"]],
        )
