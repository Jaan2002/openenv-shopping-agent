from openenv.core.env_server import Action, Observation, State


class TaskAction(Action):
    product: str


class TaskObservation(Observation):
    query: str


class TaskState(State):
    pass
