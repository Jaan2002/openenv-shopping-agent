# Copyright (c) Meta-style OpenEnv layout — HTTP shopping environment client.

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import ShoppingAction, ShoppingObservation


class ShoppingEnvClient(EnvClient[ShoppingAction, ShoppingObservation, State]):
    """
    WebSocket client for the shopping environment (standard OpenEnv layout).

    For quick local checks over HTTP, use requests against /reset and /step instead.
    """

    def _step_payload(self, action: ShoppingAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ShoppingObservation]:
        obs_data = payload.get("observation") or {}
        observation = ShoppingObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State.model_validate(payload)
