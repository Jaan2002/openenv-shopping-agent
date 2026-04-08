from models import Observation, Action, Reward, Product
from tasks import tasks


class ShoppingEnv:
    def __init__(self):
        self.current_task = tasks[0]  

    def reset(self):
        
        self.current_task = tasks[0]

        products = [Product(**p) for p in self.current_task["products"]]

        return Observation(
            category=self.current_task["category"],  
            user_need=self.current_task["user_need"],
            budget=self.current_task["budget"],
            priority=self.current_task["priority"],
            products=products
        )

    def step(self, action: Action):
        task = self.current_task
        products = task["products"]

        # Find selected product
        selected_product = None
        for p in products:
            if p["name"] == action.action_type:
                selected_product = p
                break

        score = 0.5  # base score

        if selected_product:
            # Budget check
            if selected_product["price"] <= task["budget"]:
                score += 0.2
            else:
                score -= 0.2

            # Priority scoring
            if task["priority"] == "price":
                best = min(products, key=lambda x: x["price"])
                if selected_product["name"] == best["name"]:
                    score += 0.2

            elif task["priority"] == "rating":
                best = max(products, key=lambda x: x["rating"])
                if selected_product["name"] == best["name"]:
                    score += 0.2

            elif task["priority"] == "battery":
                best = max(products, key=lambda x: x["battery"])
                if selected_product["name"] == best["name"]:
                    score += 0.2
        else:
            score = 0.1  # fallback for invalid action

        
        score = max(0.1, min(0.9, score))

        done = True
        info = {}

        observation = Observation(
            category=task["category"],  
            user_need=task["user_need"],
            budget=task["budget"],
            priority=task["priority"],
            products=[Product(**p) for p in products]
        )

        reward = Reward(score=round(score, 2))

        return observation, reward, done, info

    def state(self):
        return {"status": "running"}
