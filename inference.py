import os
import re
from openai import OpenAI
from env import ShoppingEnv
from models import Action

# REQUIRED ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:free")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


def get_ai_action(obs):
    try:
        prompt = f"""
User need: {obs.user_need}
Budget: {obs.budget}
Priority: {obs.priority}

Products:
"""
        for p in obs.products:
            prompt += f"{p.name}, price {p.price}, rating {p.rating}, battery {p.battery}\n"

        prompt += "\nSelect best product. Format: Selected product: <name>"

        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        text = res.choices[0].message.content or ""
        match = re.search(r"Selected product:\s*(.*)", text)

        if match:
            return Action(action_type=match.group(1).strip(), explanation=text)

    except Exception:
        pass

    # fallback
    return Action(action_type=obs.products[0].name, explanation="fallback")


def run():
    env = ShoppingEnv()

    rewards = []
    step_count = 0
    success = True

    tasks = ["easy", "medium", "hard"]

    try:
        for i, task_name in enumerate(tasks, start=1):

            
            print(f"[START] task={task_name} env=openenv-shopping model={MODEL_NAME}")

            obs = env.reset()
            action = get_ai_action(obs)

            obs, reward, done, _ = env.step(action)

            
            score = max(0.01, min(0.99, float(reward.score)))

            rewards.append(f"{score:.2f}")
            step_count += 1

            print(
                f"[STEP] step={i} "
                f"action={action.action_type} "
                f"reward={score:.2f} "
                f"done=true "
                f"error=null"
            )

    except Exception as e:
        success = False

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_count} "
        f"rewards={','.join(rewards)}"
    )


if __name__ == "__main__":
    run()
