"""Best-training-reward and replay-buffer persistence callback from notebook cell 11."""

import os
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, buffer_freq: int = 40000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.buffer_freq = buffer_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.buffer_path = os.path.join(log_dir, "replay_buffer")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Save replay buffer every N steps
        if self.num_timesteps % self.buffer_freq == 0:
            buffer_file = f"{self.buffer_path}"
            if self.verbose >= 1:
                print(f"Saving replay buffer to {buffer_file}")
            self.model.save_replay_buffer(buffer_file)

        # Check best model
        if self.n_calls % self.check_freq == 0:
            monitor_file = os.path.join(self.log_dir, "monitor.csv")
            if os.path.exists(monitor_file):
                results = pd.read_csv(monitor_file, skiprows=1)
                returns = results["r"].to_numpy(dtype=float)
                if len(returns) > 0:
                    mean_reward = np.mean(returns[-50:])

                    if self.verbose >= 1:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - "
                              f"Last mean reward per episode: {mean_reward:.2f}")

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose >= 1:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)

        return True
