import os
import re
from openai import OpenAI
from env import ShoppingEnv
from models import Action

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

        return Action(action_type=selected, explanation=output)

    except Exception:
        return Action(
            action_type=obs.products[0].name,
            explanation="Fallback decision"
        )


def run():
    try:
        env = ShoppingEnv()
        rewards = []
        step_count = 0

        print(f"[START] task=shopping env=openenv-shopping model={MODEL_NAME}")

        for i in range(3):
            try:
                obs = env.reset()

                action = get_ai_action(obs)

                obs, reward, done, info = env.step(action)

                rewards.append(f"{reward.score:.2f}")
                step_count += 1

                print(
                    f"[STEP] step={step_count} "
                    f"action={action.action_type} "
                    f"reward={reward.score:.2f} "
                    f"done={str(done).lower()} "
                    f"error=null"
                )

            except Exception as step_error:
                print(
                    f"[STEP] step={step_count} "
                    f"action=error "
                    f"reward=0.00 "
                    f"done=true "
                    f"error={str(step_error)}"
                )
                break

        success = True if len(rewards) > 0 else False

        print(
            f"[END] success={str(success).lower()} "
            f"steps={step_count} "
            f"rewards={','.join(rewards)}"
        )

    except Exception as e:
        print(f"[END] success=false steps=0 rewards= error={str(e)}")


if __name__ == "__main__":
    run()
