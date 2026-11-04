"""Evaluate SAC with notebook-derived battery dynamics, not thesis-derived claims.

Notebook cell 26 forces the cooling command to 0.35. To reproduce that choice,
pass --cooling-command 0.35. By default, the policy controls cooling.
"""

import argparse
import csv
from pathlib import Path
from statistics import mean

import numpy as np


def evaluate(model_mat: Path, model_path: Path, episodes: int, seed: int,
             cooling_command: float | None, output_csv: Path | None, n_cells: int):
    from stable_baselines3 import SAC
    from .env import LIBPackEnv
    env = LIBPackEnv(model=str(model_mat), number_of_cells=n_cells, T=25, dt=1)
    model = SAC.load(str(model_path), env=env, device='auto')
    if cooling_command is not None and not 0 <= cooling_command <= 1:
        raise ValueError('cooling command must be between 0 and 1')
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        reward_sum, peak_temp, peak_voltage, violations, steps = 0., -float('inf'), -float('inf'), False, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).copy()
            if cooling_command is not None:
                action[-1] = cooling_command
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            peak_temp = max(peak_temp, float(np.max(info['temperature'])))
            peak_voltage = max(peak_voltage, float(np.max(info['voltage'])))
            violations |= bool(info['safety_violation_ever'])
            steps += 1
            if terminated or truncated:
                break
        # Original notebook flattens a (3, N) state via .T.reshape(-1).
        soc = obs.reshape(n_cells, 3)[:, 0]
        rows.append({
            'episode': episode + 1, 'duration_s': steps * env.dt,
            'reached_mean_target': bool(info['is_done']),
            'completed_without_safety_violation': bool(info['success']),
            'had_safety_violation': violations,
            'final_soc_min': float(np.min(soc)),
            'final_soc_mean': float(np.mean(soc)),
            'final_soc_spread': float(np.ptp(soc)),
            'peak_voltage_v': peak_voltage, 'peak_core_temperature_c': peak_temp,
            'total_reward': reward_sum,
        })
    env.close()
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-mat', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--n-cells', type=int, default=4)
    parser.add_argument('--cooling-command', type=float, default=None)
    parser.add_argument('--output-csv', type=Path, default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.n_cells < 2:
        parser.error('episodes must be positive and n-cells >= 2')
    if not args.model_mat.is_file() or not args.model_path.is_file():
        parser.error('Both original .mat battery file and checkpoint file must exist')
    rows = evaluate(args.model_mat, args.model_path, args.episodes, args.seed,
                    args.cooling_command, args.output_csv, args.n_cells)
    for row in rows:
        print(row)
    print(f"Safety-compliant completions: {sum(r['completed_without_safety_violation'] for r in rows)}/{len(rows)}")
    print(f"Mean duration: {mean(r['duration_s'] for r in rows):.1f} seconds")
    print('These newly measured results do NOT replace thesis results in the README.')


if __name__ == '__main__':
    main()
