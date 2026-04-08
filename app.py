from fastapi import FastAPI
from env import ShoppingEnv
from models import Action

app = FastAPI()
env = ShoppingEnv()

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: dict):
    act = Action(**action)
    obs, reward, done, info = env.step(act)
    return {
        "observation": obs.dict(),
        "reward": reward.score,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return {"status": "running"}