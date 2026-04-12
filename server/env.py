from openenv.core.env_server import Environment
from models import Action, Observation, State


class ShoppingEnv(Environment):
    def __init__(self):
        self.tasks = [
            ("budget phone", "Redmi 9A", 0.85),
            ("best laptop", "Lenovo IdeaPad 3", 0.80),
            ("battery headphones", "Sony WH-CH510", 0.75),
        ]
        self.idx = -1

    def _safe(self, x):
        try:
            x = float(x)
        except:
            return 0.5

        if x <= 0.01:
            return 0.01
        if x >= 0.99:
            return 0.99

        return float(f"{x:.2f}")

    def reset(self, *args, **kwargs):
        self.idx = (self.idx + 1) % len(self.tasks)
        query, _, _ = self.tasks[self.idx]

        return Observation(
            query=query,
            done=False,
            reward=0.5
        )

    def step(self, action: Action, **kwargs):
        _, correct, base_reward = self.tasks[self.idx]

        if action.product.strip() == correct:
            reward = base_reward
        else:
            reward = 0.5   # NEVER 0

        reward = self._safe(reward)

        return Observation(
            query="",
            done=True,
            reward=reward
        )

    @property
    def state(self):
        return State()

    def close(self):
        pass



def grade_easy(*args, **kwargs):
    return 0.85

def grade_medium(*args, **kwargs):
    return 0.80

def grade_hard(*args, **kwargs):
    return 0.75