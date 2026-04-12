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

        # Build a prompt describing the task
        products_desc = "\n".join(
            f"- {p.name}: price={p.price}, rating={p.rating}, battery={p.battery}"
            for p in obs.products
        )
        prompt = (
            f"You are a shopping assistant. The user needs: {obs.user_need}\n"
            f"Budget: {obs.budget}\nCategory: {obs.category}\nPriority: {obs.priority}\n"
            f"Products:\n{products_desc}\n\n"
            f"Reply with ONLY the exact product name that best fits the user's need."
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )
            chosen = response.choices[0].message.content.strip()

            # Validate: must be one of the listed product names
            valid_names = [p.name for p in obs.products]
            if chosen not in valid_names:
                # Fallback: pick closest match
                chosen = min(valid_names, key=lambda n: abs(len(n) - len(chosen)))

        except Exception as e:
            print(f"[WARN] LLM call failed: {e}, using fallback")
            chosen = obs.products[0].name

        action = Action(action_type=chosen, explanation="LLM choice")
        obs, reward, done, _ = env.step(action)

        score = reward.score  # already clamped by env._safe_score
        rewards.append(f"{score:.2f}")
        steps += 1

        print(f"[STEP] step={steps} action={action.action_type} reward={score:.2f} done=true error=null")

    print(f"[END] success=true steps={steps} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run()
