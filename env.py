"""
ShoppingEnv — OpenEnv-compliant AI Shopping Assistant Environment
=================================================================
Each episode: reset() → step(action) → reward
Index only advances AFTER step(), not in reset().
"""

from __future__ import annotations

from typing import ClassVar, List, Any

from models import Observation, Action, Reward, Product
from tasks import tasks


class ShoppingEnv:
    """
    OpenEnv environment for multi-criteria AI shopping decisions.

    Correct episode flow
    --------------------
    obs = env.reset()        # loads current task, does NOT advance index
    obs, reward, done, info = env.step(action)  # scores + advances index

    Score range
    -----------
    All reward scores are strictly in (0.0, 1.0) — never 0.0 or 1.0.
    """

    
    TASKS: ClassVar[List[Any]] = tasks

    def __init__(self) -> None:
        self.current_task_index: int = 0
        self.current_task: dict = tasks[0]
        self._episode_count: int = 0


    def reset(self) -> Observation:
        """
        Load the current task and return its observation.
        Does NOT advance the task index — that happens in step().
        """
        if self.current_task_index >= len(tasks):
            self.current_task_index = 0

        self.current_task = tasks[self.current_task_index]
        self._episode_count += 1

        return self._build_observation(self.current_task)

   

    @staticmethod
    def compute_reward(task: dict, action: Action) -> tuple[float, dict]:
        """
        Deterministic reward for (task, action). Used by step() and by POST /grader.
        Does not mutate environment state.
        """
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

    def grade(self, task_name: str, action: Action) -> tuple[float, dict]:
        """Run grader for a named task without advancing current_task_index."""
        for t in tasks:
            if t.get("name") == task_name:
                return self.compute_reward(t, action)
        raise ValueError(f"Unknown task_name: {task_name}")

    def step(self, action: Action):
        """
        Execute one action, compute reward, then advance to next task.

        Parameters
        ----------
        action : Action
            Agent's product selection (action.action_type = product name).

        Returns
        -------
        observation : Observation
        reward      : Reward  — score strictly in (0.0, 1.0)
        done        : bool    — always True (single-step episodes)
        info        : dict    — diagnostics
        """
        task = self.current_task
        score, info = self.compute_reward(task, action)

        self.current_task_index += 1

        return (
            self._build_observation(task),
            Reward(score=score),
            True,
            info,
        )


    def state(self) -> dict:
        return {
            "status": "running",
            "episode_count": self._episode_count,
            "current_task": self.current_task.get("name"),
            "total_tasks": len(tasks),
        }



    @staticmethod
    def _build_observation(task: dict) -> Observation:
        return Observation(
            category=task["category"],
            user_need=task["user_need"],
            budget=task["budget"],
            priority=task["priority"],
            products=[Product(**p) for p in task["products"]],
        )
