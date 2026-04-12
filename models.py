from pydantic import BaseModel

class Action(BaseModel):
    product: str

class Observation(BaseModel):
    query: str
    done: bool
    reward: float

class State(BaseModel):
    pass
