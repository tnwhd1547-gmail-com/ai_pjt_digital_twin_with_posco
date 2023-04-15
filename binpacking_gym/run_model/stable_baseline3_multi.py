"""
torch, Keras등 일반 모델 러닝을 위해
Env 돌리기
"""
import sys
sys.path.append('/home/piai/workspace/Reinforcement-Learning-2d-binpacking/binpacking_gym')

import numpy as np
import gym
import binpacking_posco
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

"""
gym.make()에 들어갈 Parameter들 꼭 넣어주어야함 !
"""


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, ct2_threshold=20, print_Map=False)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    
    # Running
    env_id = 'binpacking_posco-v2'
    num_cpu = 8
    
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    #model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log='./tensorboard/')
    model = PPO.load('./model/ppo_model2', env = env, tensorboard_log='./tensorboard/')
    
    #callback_on_best = StopTrainingOnRewardThreshold(max_no_improvement_evals=100, verbose=1)
    eval_callback = EvalCallback(eval_env=gym.make(env_id, ct2_threshold=20, print_Map=False),
                                 eval_freq=100,
                                 render=False,
                                 log_path = './log/',
                                 verbose=1)
    model.learn(total_timesteps=int(2e7), callback=eval_callback,
                log_interval=10, progress_bar=True)
    
    # Save
    model.save('./model/ppo_model3.zip')
    
    
    # # 모델 결과 Rendering
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()