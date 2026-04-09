from fastapi import FastAPI
from env import ShoppingEnv
from models import Action
from inference import run

app = FastAPI()
env = ShoppingEnv()


@app.get("/")
def home():
    return {"message": "Shopping AI Env Running"}


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
    return env.state()


@app.on_event("startup")
def run_inference_on_start():
    run()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
