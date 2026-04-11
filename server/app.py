from fastapi import FastAPI
from env import ShoppingEnv
from models import Action

app = FastAPI()
env = ShoppingEnv()

@app.get("/")
def home():
    return {"message": "Running"}

@app.post("/reset")
def reset():
    return env.reset().model_dump()

@app.post("/step")
def step(action: dict):
    act = Action(**action)
    obs, reward, done, _ = env.step(act)

    return {
        "observation": obs.model_dump(),
        "reward": reward.score,
        "done": done
    }

@app.get("/state")
def state():
    return env.state()
def main():
    import os
    import uvicorn

    port = int(os.getenv("PORT", "7860")) 
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
