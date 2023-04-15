import gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import binpacking_posco

def get_action_mask(env):
    """
    현재 상태에서 가능한 Action ndarray
    현재 state에서 모든 action이 boolean으로 표현
    """
    return env.mask_action()


env = gym.make('binpacking_posco-v00', print_Map=False)  # Initialize env
env = ActionMasker(env, get_action_mask)  # Wrap to enable masking

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log='C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/tensorboard/my_maskppo')
model.learn(2*int(5e6), progress_bar=True)
model.save('C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/model/my_mask_ppo_v00')
# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
# model.predict(observation, action_masks=valid_action_array)