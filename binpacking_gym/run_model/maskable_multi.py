import sys
sys.path.append('/home/piai/workspace/Reinforcement-Learning-2d-binpacking/binpacking_gym')

import numpy as np
import gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import binpacking_posco

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env = ActionMasker(env, get_action_mask)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_action_mask(env):
    """
    현재 상태에서 가능한 Action ndarray
    현재 state에서 모든 action이 boolean으로 표현
    """
    return env.mask_action()

if __name__ == '__main__':
    env_id = 'binpacking_posco-v3'
    num_cpu = 2

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    
    #model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log='./tensorboard/maskppo1')
    model = MaskablePPO.load('./model/mask_ppo1', env=env, tensorboard_log='./tensorboard/maskppo1')
    model.learn(int(2e7), progress_bar=True, log_interval=10)
    model.save('./model/mask_ppo2')
#env = gym.make('binpacking_posco-v3', print_Map=False)
#env = ActionMasker(env, get_action_mask)

# model.learn(int(2e6), progress_bar=True)

# model.save('./model/mask_ppo1')

# model = MaskablePPO.load('./model/mask_ppo1', env)
# model.get_env()
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
# print (mean_reward, std_reward)

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     print (rewards)