"""
ShoppingEnv — OpenEnv-compliant AI Shopping Assistant Environment
=================================================================
Models a real-world multi-criteria product recommendation task.
An agent receives a shopping scenario (user need, budget, priority)
and must select the best product from a candidate list.

Reward is shaped to provide a dense, informative signal:
  - Grader match (optimal product selected)  → high base score
  - Budget compliance bonus/penalty
  - Priority alignment bonus (price / rating / battery)
  - Partial credit for near-optimal choices

All scores are strictly within (0.0, 1.0) exclusive.
"""

from __future__ import annotations

from models import Observation, Action, Reward, Product
from tasks import tasks


class ShoppingEnv:
    """
    OpenEnv environment for multi-criteria AI shopping decisions.

    Episodes
    --------
    Each episode corresponds to one shopping task. The agent receives
    an Observation, picks one product via Action, and receives a Reward.

    State transitions
    -----------------
    reset() → Observation  (start new episode)
    step(action) → (Observation, Reward, done, info)

    Score range
    -----------
    All reward scores are strictly in (0.0, 1.0) — never 0.0 or 1.0.
    """

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self.current_task_index: int = 0
        self.current_task: dict = tasks[0]
        self._episode_count: int = 0

    def reset(self) -> Observation:
        """
        Start a new episode.

        Cycles through all tasks in order; wraps around after the last task.

        Returns
        -------
        Observation
            The initial observation for the new episode.
        """
        if self.current_task_index >= len(tasks):
            self.current_task_index = 0

        self.current_task = tasks[self.current_task_index]
        self.current_task_index += 1
        self._episode_count += 1

        return self._build_observation(self.current_task)

    # ------------------------------------------------------------------ #
    #  Core step                                                           #
    # ------------------------------------------------------------------ #

    def step(self, action: Action):
        """
        Execute one action and return the transition tuple.

        Parameters
        ----------
        action : Action
            Agent's product selection (action.action_type = product name).

        Returns
        -------
        observation : Observation
            Post-step observation (same task, for logging / consistency).
        reward : Reward
            Score strictly in (0.0, 1.0).
        done : bool
            Always True — each task is a single-step episode.
        info : dict
            Diagnostic info (optimal product, score breakdown).
        """
        task = self.current_task
        products = task["products"]
        grader = task.get("grader", {})

        # ── Find selected product ──────────────────────────────────────
        selected = None
        for p in products:
            if p["name"].strip().lower() == action.action_type.strip().lower():
                selected = p
                break

        # ── Base score from grader ────────────────────────────────────
        if selected is None:
            # Invalid / unrecognised product name — minimal score
            score = 0.15
            breakdown = {"reason": "product_not_found", "base": 0.15}
        else:
            optimal_name = grader.get("target", task.get("optimal", ""))
            is_optimal = selected["name"] == optimal_name

            if is_optimal:
                score = grader.get("score_if_correct", 0.75)
                breakdown = {"reason": "optimal_match", "base": score}
            else:
                score = grader.get("score_if_wrong", 0.30)
                breakdown = {"reason": "suboptimal", "base": score}

            # ── Dense reward shaping ──────────────────────────────────
            bonus = 0.0

            # Budget compliance
            if selected["price"] <= task["budget"]:
                bonus += 0.05
                breakdown["budget_bonus"] = 0.05
            else:
                bonus -= 0.08
                breakdown["budget_penalty"] = -0.08

            # Priority alignment
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

        # ── Clamp strictly within (0.0, 1.0) ─────────────────────────
        score = max(0.12, min(0.88, score))
        score = round(score, 2)

        info = {
            "optimal": task.get("optimal"),
            "selected": action.action_type,
            "score_breakdown": breakdown,
            "task_name": task.get("name"),
        }

        return (
            self._build_observation(task),
            Reward(score=score),
            True,   # done — single-step episodes
            info,
        )

    # ------------------------------------------------------------------ #
    #  State & helpers                                                     #
    # ------------------------------------------------------------------ #

    def state(self) -> dict:
        """Return lightweight environment state for inspection."""
        return {
            "status": "running",
            "episode_count": self._episode_count,
            "current_task": self.current_task.get("name"),
            "total_tasks": len(tasks),
        }

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_observation(task: dict) -> Observation:
        return Observation(
            category=task["category"],
            user_need=task["user_need"],
            budget=task["budget"],
            priority=task["priority"],
            products=[Product(**p) for p in task["products"]],
        )
