from models import Observation, Action, Reward, Product
from tasks import tasks


class ShoppingEnv:
    def __init__(self):
        self.current_task_index = 0
        self.current_task = None

    def reset(self):
      self.current_task = tasks[self.current_task_index]

      products = [Product(**p) for p in self.current_task["products"]]

      return Observation(
          user_need=self.current_task["user_need"],
          budget=self.current_task["budget"],
          priority=self.current_task["priority"],
          category=self.current_task["category"],  
          products=products
       )

    def step(self, action: Action):
        selected = action.action_type
        optimal = self.current_task["optimal"]
        products = self.current_task["products"]

        score = 0.0

        # Find selected product
        selected_product = None
        for p in products:
            if p["name"] == selected:
                selected_product = p
                break

        # Grading Logic
        if selected == optimal:
            score += 1.0
        elif selected_product:
            score += 0.5
        else:
            score -= 0.5

        # Budget penalty (real-world constraint)
        if selected_product and selected_product["price"] > self.current_task["budget"]:
            score -= 0.3

        # Explanation bonus
        if action.explanation and len(action.explanation) > 15:
            score += 0.2

        # Clamp score (0 → 1)
        score = max(0.0, min(score, 1.0))

        reward = Reward(score=score)
        done = True

        # Move to next task
        self.current_task_index += 1
        if self.current_task_index >= len(tasks):
            self.current_task_index = 0

        return self.reset(), reward, done, {}

    def state(self):
        return self.current_task