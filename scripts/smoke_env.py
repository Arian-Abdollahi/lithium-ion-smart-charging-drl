"""Smoke-check the notebook environment with a user-supplied PANmodel.mat."""

import argparse
from pathlib import Path


def main():
    from battery_smart_charging.env import LIBPackEnv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-mat', type=Path, required=True)
    args = parser.parse_args()
    env = LIBPackEnv(model=args.model_mat)
    obs, _ = env.reset(seed=42)
    state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    print(f'Observation {obs.shape} -> {state.shape}; reward={reward:.4f}; stop={terminated or truncated}')
    env.close()


if __name__ == '__main__':
    main()
