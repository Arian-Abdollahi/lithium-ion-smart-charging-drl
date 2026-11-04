"""Train SAC on the original notebook-derived battery environment.

The original Kaggle experiment trained over multiple sessions; those private
checkpoints and replay buffers are not part of the uploaded notebook.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-mat', type=Path, required=True, help='Original PANmodel.mat battery model')
    parser.add_argument('--policy', choices=('attention', 'mlp'), default='attention')
    parser.add_argument('--timesteps', type=int, default=1_700_000)
    parser.add_argument('--n-cells', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--log-dir', type=Path, default=Path('runs/attention'))
    parser.add_argument('--resume-model', type=Path, default=None)
    parser.add_argument('--resume-replay-buffer', type=Path, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--check-env', action='store_true')
    args = parser.parse_args()
    if args.timesteps < 1 or args.n_cells < 2:
        parser.error('timesteps must be positive and n-cells must be at least two')
    if args.resume_replay_buffer is not None and args.resume_model is None:
        parser.error('--resume-replay-buffer requires --resume-model')
    if not args.model_mat.is_file():
        parser.error(f'Missing original battery model: {args.model_mat}')

    import torch as th
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_checker import check_env
    from .callbacks import SaveOnBestTrainingRewardCallback
    from .env import LIBPackEnv
    from .features import SetTransformerExtractor

    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = LIBPackEnv(model=str(args.model_mat), number_of_cells=args.n_cells, T=25, dt=1)
    if args.check_env:
        check_env(env, warn=True)
    monitored_env = Monitor(
        env, str(args.log_dir / 'monitor.csv'),
        info_keywords=('is_done', 'soc_min', 'soc_mean', 'soc_dev', 'gate', 'success'),
        override_existing=args.resume_model is None,
    )
    lr = args.learning_rate if args.learning_rate is not None else (3e-4 if args.resume_model else 1e-3)
    if args.resume_model is not None:
        model = SAC.load(str(args.resume_model), env=monitored_env, device=args.device)
        model.learning_rate = lr
        model.lr_schedule = lambda _: lr
        for optimizer in (model.actor.optimizer, model.critic.optimizer):
            for group in optimizer.param_groups:
                group['lr'] = lr
        if model.ent_coef_optimizer is not None:
            for group in model.ent_coef_optimizer.param_groups:
                group['lr'] = lr
        if args.resume_replay_buffer:
            model.load_replay_buffer(str(args.resume_replay_buffer))
    else:
        policy_kwargs = {'net_arch': dict(pi=[128, 64], qf=[128, 64]), 'activation_fn': th.nn.ReLU}
        if args.policy == 'attention':
            policy_kwargs.update({
                'features_extractor_class': SetTransformerExtractor,
                'features_extractor_kwargs': dict(n_cells=args.n_cells, cell_feat_dim=3,
                                                  d_model=32, n_heads=4, latent_dim=64),
            })
        model = SAC('MlpPolicy', monitored_env, verbose=1, gamma=0.95,
                    tensorboard_log=str(args.log_dir / 'tensorboard'),
                    policy_kwargs=policy_kwargs, learning_rate=lr, device=args.device,
                    ent_coef=0.01, buffer_size=int(1e6), batch_size=1024, seed=args.seed)
    callbacks = [
        SaveOnBestTrainingRewardCallback(check_freq=20_000, log_dir=str(args.log_dir)),
        CheckpointCallback(save_freq=50_000, save_path=str(args.log_dir / 'checkpoints'),
                           name_prefix='battery_rl_model'),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=False, reset_num_timesteps=args.resume_model is None)
    model.save(str(args.log_dir / 'final_model'))
    model.save_replay_buffer(str(args.log_dir / 'replay_buffer.pkl'))
    monitored_env.close()
    print(f'New experiment artifacts saved under {args.log_dir}.')
    print('The thesis PDF results are not derived from this new run.')


if __name__ == '__main__':
    main()
