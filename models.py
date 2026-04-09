from typing import Any, List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field


class Product(BaseModel):
    name: str
    price: float
    rating: float
    battery: int


class ShoppingAction(Action):
    """Agent action: choose a product by name."""

    action_type: str = Field(..., description="Selected product name")
    explanation: str = Field(default="", description="Reasoning / model output")


class ShoppingObservation(Observation):
    """What the agent sees for the current shopping task."""

    user_need: str = Field(..., description="Natural-language need")
    budget: float = Field(..., description="Maximum budget")
    category: str = Field(..., description="Product category")
    priority: str = Field(..., description="Optimization priority (price, rating, battery)")
    products: List[Product] = Field(..., description="Candidate products")


class GradeRequest(BaseModel):
    """POST /grader — score a (task_name, action) pair without advancing the queue."""

    model_config = ConfigDict(extra="forbid")

    task_name: str
    action: ShoppingAction
