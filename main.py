import argparse
import pickle

import numpy as np
import wandb, warnings
import torch
from config import args_khazad_dum_varibad
from config.mujoco import args_cheetah_vel_rl2, args_cheetah_vel_varibad, args_cheetah_mass_varibad, \
    args_cheetah_body_varibad, args_ant_goal_rl2, args_ant_goal_varibad, args_ant_mass_varibad, \
    args_humanoid_vel_varibad, args_humanoid_mass_varibad, args_humanoid_body_varibad
from environments.parallel_envs import make_vec_envs
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.logger import configure
# from cuquantum_ppo.custom_ppo import CuQuantumPPO
# from cuquantum_ppo.custom_callback import CuQuantumCallback
# from muzero.muzero import MuZero
import logging
from stable_baselines3.common import evaluation

import os


def generate_exp_label(args):
    if not args.exp_label:
        if args.oracle:
            method = 'oracle'  # 'Oracle'
        elif args.cem == 0 and args.tail == 0:
            method = 'varibad'  # 'VariBAD'
        elif args.cem == 1:
            method = 'cembad'  # 'RoML'
        elif args.tail == 1:
            method = 'cvrbad'  # 'CVaR_MRL'
        else:
            raise ValueError(args.cem, args.tail)

        env_name_map = {
            'KhazadDum-v0': 'kd',
            'HalfCheetahVel-v0': 'hcv',
            'HalfCheetahMass-v0': 'hcm',
            'HalfCheetahBody-v0': 'hcb',
            'HumanoidVel-v0': 'humv',
            'HumanoidMass-v0': 'humm',
            'HumanoidBody-v0': 'humb',
            'AntMass-v0': 'antm',
        }
        env_name = env_name_map[args.env_name]

        args.exp_label = f'{env_name}_{method}'

        if isinstance(args.seed, (tuple, list)):
            args.seed = args.seed[0]

    try:
        if args.exp_suffix:
            args.exp_label = f'{args.exp_label}_{args.exp_suffix}'
    except:
        warnings.warn(f'Missing attribute args.exp_label')

    return args


# os.environ["WANDB_MODE"] = "dryrun"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_varibad')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    if env == 'khazad_dum_varibad':
        args = args_khazad_dum_varibad.get_args(rest_args)

    # --- MUJOCO ---

    # - Cheetah -
    elif env == 'cheetah_vel_varibad':
        args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == 'cheetah_vel_rl2':
        args = args_cheetah_vel_rl2.get_args(rest_args)
    elif env == 'cheetah_mass_varibad':
        args = args_cheetah_mass_varibad.get_args(rest_args)
    elif env == 'cheetah_body_varibad':
        args = args_cheetah_body_varibad.get_args(rest_args)
    #
    # - Humanoid -
    elif env == 'humanoid_vel_varibad':
        args = args_humanoid_vel_varibad.get_args(rest_args)
    elif env == 'humanoid_mass_varibad':
        args = args_humanoid_mass_varibad.get_args(rest_args)
    elif env == 'humanoid_body_varibad':
        args = args_humanoid_body_varibad.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    args = generate_exp_label(args)
    # config = setup_experiment(project='RLOptimizer', enable_wandb_update=True)
    env = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                        gamma=args.policy_gamma, device='cpu',
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        tasks=None,
                        )

    ngc_run = os.path.isdir('/ws')
    if ngc_run:
        ngc_dir = '/result/wandb/'  # args.ngc_path
        os.makedirs(ngc_dir, exist_ok=True)
        logging.info('NGC run detected. Setting path to workspace: {}'.format(ngc_dir))
        wandb.init(project="roml", sync_tensorboard=True, config=args, dir=ngc_dir)
    else:
        wandb.init(project="roml", sync_tensorboard=True, config=args)
    logger = configure(wandb.run.dir, ["stdout", "tensorboard"])
    # n_steps = config['ppo']['steps_per_epoch_per_env']
    # total_timesteps = config['ppo']['total_steps']
    train_algo = PPO  # if learning_method == 'ppo' else CuQuantumPPO
    # callback = CustomEvalCallback(config, n_eval_episodes=config['eval']['n_eval_episodes'],
    #                               eval_freq=config['eval']['eval_freq'])
    model = PPO(policy=MlpPolicy, env=env, tensorboard_log='./runs', seed=args.seed,
                env_name=args.env_name, cem_alpha=args.alpha if args.cem else 0)
    model.set_logger(logger)
    model.learn(args.num_frames)

    ############ TEST ###########
    test_env = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                             gamma=1.0, device='cpu', episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None, tasks=None,
                             )
    rets, lens, test_tasks = evaluation.evaluate_policy(model, test_env, return_episode_rewards=True,
                                                        n_eval_episodes=1008)
    output_file = os.path.join(wandb.run.dir, 'test_res.pkl')
    with open(output_file, 'wb') as fd:
        pickle.dump((rets, lens, test_tasks), fd)
