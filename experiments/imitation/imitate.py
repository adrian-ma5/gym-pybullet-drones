
import os
import time
from datetime import datetime
from sys import platform
import argparse
import subprocess
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.cmd_util import make_vec_env # Module cmd_util will be renamed to env_util https://github.com/DLR-RM/stable-baselines3/pull/197
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, \
    StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement

from gym_pybullet_drones.envs.single_agent_rl.UpDownAviary import UpDownAviary
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

EPISODE_REWARD_THRESHOLD = -0 # Upperbound: rewards are always negative, but non-zero
"""float: Reward threshold to halt the script."""

#DEFAULT_ENV = 'hover'
DEFAULT_ENV = 'updown'
DEFAULT_ALGO = 'ddpg'   # 'ppo' # choices=['a2c', 'ppo', 'sac', 'td3', 'ddpg']
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('pid') # ActionType('one_d_rpm')
DEFAULT_CPU = 1
DEFAULT_STEPS = 20000   # 50000 #   # 1500000 #1000000 4hrs #35000
DEFAULT_OUTPUT_FOLDER = 'results'
AGGR_PHY_STEPS = 5

env = DEFAULT_ENV
algo = DEFAULT_ALGO
obs = DEFAULT_OBS
act = DEFAULT_ACT
cpu = DEFAULT_CPU
steps = DEFAULT_STEPS
output_folder = DEFAULT_OUTPUT_FOLDER

env = DEFAULT_ENV
env_name = env + "-aviary-v0"
sa_env_kwargs = dict(aggregate_phy_steps=AGGR_PHY_STEPS, obs=obs, act=act)

filename = os.path.join(output_folder,
                        'save-' + env + '-' + algo + '-' + obs.value + '-' + act.value + '-' + datetime.now().strftime(
                            "%m.%d.%Y_%H.%M.%S"))
if not os.path.exists(filename):
    os.makedirs(filename + '/')

train_env = make_vec_env(UpDownAviary,
                         env_kwargs=sa_env_kwargs,
                         n_envs=cpu,
                         seed=0
                         )
offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=[512, 512, 256, 128]
                        )

if algo == 'ddpg':
    model = DDPG(td3ddpgMlpPolicy,
                 train_env,
                 policy_kwargs=offpolicy_kwargs,
                 #tensorboard_log=filename + '/tb/',
                 verbose=1
                 ) if obs == ObservationType.KIN else DDPG(td3ddpgCnnPolicy,
                                                           train_env,
                                                           policy_kwargs=offpolicy_kwargs,
                                                           #tensorboard_log=filename + '/tb/',
                                                           verbose=1
                                                           )

print(model)

model.learn(total_timesteps=100)


reward, _ = evaluate_policy(model, train_env, 10)
print(reward)

print(DummyVecEnv([lambda: RolloutInfoWrapper(train_env)]))

rng = np.random.default_rng()

print(rollout.make_sample_until(min_timesteps=None, min_episodes=1))

rng = np.random.default_rng()
rollouts = rollout.rollout(
    model,
    DummyVecEnv([lambda: RolloutInfoWrapper(train_env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=2),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)




# print(transitions)



