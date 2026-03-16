from __future__ import annotations

from typing import Final

import gymnasium as gym
import numpy as np

ENV_IDS: Final[list[str]] = [
    "msgym/ManipulationEnv-v1",
    "msgym/LocomotionFullEnv-v1",
    "msgym/LocomotionLegsEnv-v1",
]


def _smoke_test_env(env_id: str) -> None:
    """Create the env, reset, take a single random step, and close."""
    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        action = env.action_space.sample()
        step_out = env.step(action)
        assert len(step_out) == 5
        obs2, reward, terminated, truncated, info2 = step_out
        assert isinstance(obs2, np.ndarray)
        assert np.isscalar(reward)
        # Gymnasium may return bool or numpy.bool_ for done flags
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info2, dict)
    finally:
        env.close()


def test_all_msgym_envs_smoke() -> None:
    """Smoke-test all registered msgym environments."""
    for env_id in ENV_IDS:
        _smoke_test_env(env_id)
