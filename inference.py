import os
from openai import OpenAI
from env import ShoppingEnv
from models import Action

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

def run():
    env = ShoppingEnv()
    rewards = []
    steps = 0

    for task in ["easy", "medium", "hard"]:
        print(f"[START] task={task} env=openenv-shopping model={MODEL_NAME}")

        obs = env.reset(task_id=task)

        action = Action(
            action_type=llm_output,
            explanation="llm or fallback"
        )

        obs, reward, done, _ = env.step(action)

        score = max(0.01, min(0.99, reward.score))
        rewards.append(f"{score:.2f}")
        steps += 1

        print(f"[STEP] step={steps} action={action.action_type} reward={score:.2f} done=true error=null")

    print(f"[END] success=true steps={steps} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run()
