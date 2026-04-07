import os
import re
from openai import OpenAI
from env import ShoppingEnv
from models import Action


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

Then explain:
1. Why it is the best choice
2. Why other options are not suitable
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        output = response.choices[0].message.content

        # Robust parsing
        selected = None
        match = re.search(r"Selected product:\s*(.*)", output, re.IGNORECASE)

        if match:
            extracted_name = match.group(1).strip()
            for p in obs.products:
                if p.name.lower() in extracted_name.lower():
                    selected = p.name
                    break

        if not selected:
            for p in obs.products:
                if p.name.lower() in output.lower():
                    selected = p.name
                    break

        if not selected:
            selected = obs.products[0].name

        return Action(
            action_type=selected,
            explanation=output
        )

    except Exception as e:
        return Action(
            action_type=obs.products[0].name,
            explanation=f"Fallback due to error: {e}"
        )


def run():
    env = ShoppingEnv()
    rewards = []
    step_count = 0

    # START
    print(f"[START] task=shopping env=openenv-shopping model={MODEL_NAME}")

    try:
        for i in range(3):
            obs = env.reset()

            action = get_ai_action(obs)

            obs, reward, done, info = env.step(action)

            rewards.append(f"{reward.score:.2f}")
            step_count += 1

            error_msg = None
            if hasattr(obs, "last_action_error") and obs.last_action_error:
                error_msg = obs.last_action_error

            # STEP
            print(
                f"[STEP] step={step_count} "
                f"action={action.action_type} "
                f"reward={reward.score:.2f} "
                f"done={str(done).lower()} "
                f"error={error_msg if error_msg else 'null'}"
            )

    except Exception as e:
        print(f"[STEP] step={step_count} action=error reward=0.00 done=true error={str(e)}")

    finally:
        if hasattr(env, "close"):
            env.close()

        success = True if len(rewards) > 0 else False

        # END
        print(
            f"[END] success={str(success).lower()} "
            f"steps={step_count} "
            f"rewards={','.join(rewards)}"
        )


if __name__ == "__main__":
    run()
