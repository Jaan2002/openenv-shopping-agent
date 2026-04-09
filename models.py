from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    name: str
    price: float
    rating: float
    battery: int

class Observation(BaseModel):
    user_need: str
    budget: float
    category: str 
    priority: str   
    products: List[Product]

class Action(BaseModel):
    action_type: str
    explanation: str  

class Reward(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0)


class GradeRequest(BaseModel):
    """Body for POST /grader — deterministic grading without advancing the task queue."""

    task_name: str
    action: Action
