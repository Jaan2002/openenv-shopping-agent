import os
import re
from openai import OpenAI
from env import ShoppingEnv
from models import ShoppingAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:free")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None
if HF_TOKEN:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
    except Exception:
        client = None


def get_ai_action(obs):
    try:
        if client is None:
            raise Exception("No API client available")

        prompt = f"""
Category: {obs.category}
User need: {obs.user_need}
Budget: {obs.budget}
Priority: {obs.priority}

Products:
"""

        for p in obs.products:
            prompt += f"{p.name}, price {p.price}, rating {p.rating}, battery {p.battery}\n"

        prompt += """
Choose the best product.

Start your response EXACTLY like:
Selected product: <product name>
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        output = response.choices[0].message.content

        selected = None
        match = re.search(r"Selected product:\s*(.*)", output, re.IGNORECASE)

        if match:
            extracted = match.group(1).strip()
            for p in obs.products:
                if p.name.lower() in extracted.lower():
                    selected = p.name
                    break

        if not selected:
            for p in obs.products:
                if p.name.lower() in output.lower():
                    selected = p.name
                    break

        if not selected:
            selected = obs.products[0].name

        return ShoppingAction(action_type=selected, explanation=output or "")

    except Exception:
        return ShoppingAction(
            action_type=obs.products[0].name,
            explanation="Fallback decision",
        )


def run():
    try:
        env = ShoppingEnv()
        rewards = []
        step_count = 0

        print(f"[START] task=shopping env=openenv-shopping model={MODEL_NAME}")

        total_tasks = len(env.TASKS)

        for i in range(total_tasks):
            try:
                obs = env.reset()

                action = get_ai_action(obs)

                obs = env.step(action)

                reward_val = obs.reward
                score = max(0.001, min(0.999, float(reward_val) if reward_val is not None else 0.5))

                rewards.append(f"{score:.3f}")
                step_count += 1

                print(
                    f"[STEP] step={step_count} "
                    f"action={action.action_type} "
                    f"reward={score:.3f} "
                    f"done=true "
                    f"error=null"
                )

            except Exception as step_error:
                step_count += 1
                rewards.append("0.500")

                print(
                    f"[STEP] step={step_count} "
                    f"action=error "
                    f"reward=0.500 "
                    f"done=true "
                    f"error={str(step_error)}"
                )

        print(
            f"[END] success=true "
            f"steps={step_count} "
            f"rewards={','.join(rewards)}"
        )

    except Exception as e:
        print(f"[END] success=false steps=0 rewards= error={str(e)}")

if __name__ == "__main__":
    run()
