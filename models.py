from pydantic import BaseModel
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
    score: float