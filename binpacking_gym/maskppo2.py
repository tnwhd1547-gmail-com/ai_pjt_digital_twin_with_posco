import gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import binpacking_posco
from stable_baselines3.common.callbacks import EvalCallback
from acm import mask_fn

def get_action_mask(env):
    """
    현재 상태에서 가능한 Action ndarray
    현재 state에서 모든 action이 boolean으로 표현
    """
    return env.mask_action()

# binpacking_myposco-v2 -> 새로운 환경 v2 버전 사용시
# binpacking_posco-v00 -> 기존 환경 v0 사용시 
env = gym.make('binpacking_myposco-v2', print_Map=False)  # Initialize env
env = ActionMasker(env, get_action_mask)  # Wrap to enable masking

# my env 용
# env = ActionMasker(env, mask_fn)

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
# model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log='C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/tensorboard/my_maskppo')
# tb_writer = SummaryWriter(log_dir='./logs')
model = MaskablePPO(MaskableActorCriticPolicy, env=env, tensorboard_log='./tensorboard/maskppo_mynewv2',verbose=1)
env_id = 'binpacking_myposco-v2'
eval_callback = EvalCallback(eval_env=gym.make(env_id,  ct2_threshold=20, print_Map=False),
                    eval_freq=1000,
                    n_eval_episodes = 10, # 새로 추가 전기수 코드에 없었음
                    render=False,
                    best_model_save_path = './logs/best_model',
                    log_path = './log/results',
                    verbose=1)
# model = MaskablePPO.load('./model/mask_multi_myppo_v2_ver1', env=env, tensorboard_log='./tensorboard/my_maskppo')


model.learn(int(2e6), callback=eval_callback, progress_bar=True)
model.save('./model/my_maskppo_mynewv2')
# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
# model.predict(observation, action_masks=valid_action_array)