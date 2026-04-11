from models import Observation, Action, Reward, Product
from tasks import tasks


class ShoppingEnv:
    def __init__(self):
        self.index = 0
        self.task = tasks[0]

    def _safe_score(self, x):
        try:
            x = float(x)
        except Exception:
            x = 0.5

        
        if x <= 0.01:
            return 0.01
        if x >= 0.99:
            return 0.99

        return float(f"{x:.2f}")

    def reset(self):
        if self.index >= len(tasks):
            self.index = 0

        self.task = tasks[self.index]

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
            if self.task["name"] == "easy":
                score = 0.85
            elif self.task["name"] == "medium":
                score = 0.80
            else:
                score = 0.75
        else:
            score = 0.25

       
        score = self._safe_score(score)

        # Move to next task
        self.index += 1

        observation = Observation(
            user_need=self.task["user_need"],
            budget=self.task["budget"],
            category=self.task["category"],
            priority=self.task["priority"],
            products=[Product(**p) for p in self.task["products"]],
        )

        return (
            observation,
            Reward(score=score),
            True,
            {}
        )

    def state(self):
        return {"total_tasks": len(tasks)}
