from models import Observation, Action, Reward, Product
from tasks import tasks


class ShoppingEnv:
    def __init__(self):
        self.index = 0

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

        score = max(0.01, min(0.99, score))

        self.index += 1

        return (
            self.reset(),
            Reward(score=score),
            True,
            {}
        )

    def state(self):
        return {"total_tasks": len(tasks)}
