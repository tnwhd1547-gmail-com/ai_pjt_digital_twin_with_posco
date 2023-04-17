import sys
sys.path.append('C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym')

import numpy as np
import gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import binpacking_posco

def get_action_mask(env):
    """
    현재 상태에서 가능한 Action ndarray
    현재 state에서 모든 action이 boolean으로 표현
    """
    return env.mask_action()
# binpacking_posco-v00 사용시 
# env = gym.make('binpacking_posco-v00', print_Map=False)

env = gym.make('binpacking_myposco-v2', print_Map=False)
env = ActionMasker(env, get_action_mask)
    
#model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log='./tensorboard/maskppo1')
model = MaskablePPO.load('C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/model/mask_multi_myppo_v2_ver1', env=env, tensorboard_log='C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/tensorboard/maskppo1')
# model.learn(2*int(2e6), progress_bar=True, log_interval=10)
# model.save('C:/workspace/pba/r2dm/binpacking_gym/model/mask_ppo_v4_2')

## 아래 주석 풀었음
model.learn(int(2e5), progress_bar=True)

# model.save('./model/mask_ppo1')
model.save('C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/model/mask_ppo_2e6_v0')

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
