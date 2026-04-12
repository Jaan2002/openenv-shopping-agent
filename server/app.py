from openenv.core.env_server import create_fastapi_app
from server.env import ShoppingEnv, Action, Observation

app = create_fastapi_app(
    ShoppingEnv,
    Action,
    Observation
)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
