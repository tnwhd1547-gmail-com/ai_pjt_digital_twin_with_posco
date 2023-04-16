import gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import binpacking_posco
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
model = MaskablePPO(MaskableActorCriticPolicy, env=env, tensorboard_log='./tensorboard/multi_maskppo_myv2',verbose=1)
# model = MaskablePPO.load('C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/model/mask_ppo_v4_1', env=env, tensorboard_log='C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/tensorboard/my_maskppo')
model.learn(int(2e4), progress_bar=True)
model.save('C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/model/my_mask_ppo_myenvv2')
# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
# model.predict(observation, action_masks=valid_action_array)