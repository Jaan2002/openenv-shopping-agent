import os
import re
from openai import OpenAI
from env import ShoppingEnv
from models import Action

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def get_ai_action(obs):
    try:
        # Build prompt
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

        # API call
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        output = response.choices[0].message.content

        # Robust parsing
        selected = None

        #  Extract the "Selected product"
        match = re.search(r"Selected product:\s*(.*)", output, re.IGNORECASE)

        if match:
            extracted_name = match.group(1).strip()

            for p in obs.products:
                if p.name.lower() in extracted_name.lower():
                    selected = p.name
                    break

        
        if not selected:
            for p in obs.products:
                if f"**{p.name}**" in output:
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
        print("Fallback due to error:", e)

        return Action(
            action_type=obs.products[0].name,
            explanation="Fallback decision"
        )


def run():
    print("Initializing Smart Shopping Environment for AI Agent Evaluation...")

    env = ShoppingEnv()
    total_score=0
    
    for i in range(3):
        obs = env.reset()
        
        print(f"\nTask {i+1} ({['Easy','Medium','Hard'][i]}): {obs.category} | {obs.user_need}")
        print("User need:", obs.user_need)
        print("Products:", [p.name for p in obs.products])

        action = get_ai_action(obs)

        print("Selected Product:", action.action_type)
        print("Explanation:", action.explanation, "...")

        obs, reward, done, _ = env.step(action)

        print("Reward Score:", reward.score)

        total_score += reward.score 

    print("\nFinal Score across tasks:", total_score) 


if __name__ == "__main__":
    run()
