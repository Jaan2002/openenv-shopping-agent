from models import Observation, Action, Reward, Product
from tasks import tasks


class ShoppingEnv:
    def __init__(self):
        self.task = tasks[0]

    def _safe_score(self, x):
        x = float(x)
        return round(max(0.02, min(0.98, x)), 2)

    def reset(self, task_id=None):
        if task_id:
            for t in tasks:
                if t["id"] == task_id:
                    self.task = t
                    break

        return Observation(
            user_need=self.task["user_need"],
            budget=self.task["budget"],
            category=self.task["category"],
            priority=self.task["priority"],
            products=[Product(**p) for p in self.task["products"]],
        )

    def step(self, action: Action):
        optimal = self.task["optimal"]

        if action.action_type == optimal:
            if self.task["id"] == "easy":
                score = 0.85
            elif self.task["id"] == "medium":
                score = 0.80
            else:
                score = 0.75
        else:
            score = 0.25

        score = self._safe_score(score)

        return (
            self.reset(task_id=self.task["id"]),
            Reward(score=score),
            True,
            {}
        )

    def state(self):
        return {"total_tasks": len(tasks)}
