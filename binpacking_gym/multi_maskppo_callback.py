import gym
import numpy as np
import binpacking_posco
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
from acm import mask_fn

## callback 함수
# rollout_freq, eval_freq = 100, 100

# def rollout_callback(locals, globals):
#     if locals['self'].num_timesteps % rollout_freq == 0:
#         env = gym.make('binpacking_myposco-v2')
#         obs = env.reset()
#         done = False
#         total_reward = 0
#         while not done:
#             action, _ = locals['self'].predict(obs)
#             obs, reward, done, info = env.step(action)
#             total_reward += reward
#         print("Rollout at timestep {}: reward={}".format(locals['self'].num_timesteps, total_reward))

# def eval_callback(locals, globals):
#     if locals['self'].num_timesteps % eval_freq == 0:
#         env = gym.make('binpacking_myposco-v2')
#         eval_env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
#         mean_reward, _ = locals['self'].evaluate(eval_env, n_eval_episodes=5)
#         print("Evaluation at timestep {}: mean_reward={}".format(locals['self'].num_timesteps, mean_reward))




##### callback 함수 끝
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
        env = ActionMasker(env, mask_fn)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# env = gym.make('binpacking_posco-v00', print_Map=False)  # Initialize env
# env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept


# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time

# model.predict(observation, action_masks=valid_action_array)

if __name__ == '__main__':
    
    #  기존환경 v0 mask ppo 돌릴때 binpacking_posco-v00
    # 새로 만든 환경 v2 mask ppo binpacking_myposco-v2
    # Running
    env_id = 'binpacking_myposco-v2'
    num_envs = 8
    
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])

    #model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log='./tensorboard/')
    # 액션가면 지선생이 안씀
    
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    model = MaskablePPO(MaskableActorCriticPolicy, env=env, tensorboard_log='./tensorboard/multi_maskppo_myv2',verbose=1)
    # 기존 모델 로드 부분 주석처리함
    # model = MaskablePPO.load('C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/model/mask_multi_ppo_model_10e6', env=env, tensorboard_log='C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/tensorboard/multi_maskppo_myv2')
    
    # model = PPO(MaskableMlpPolicy, env, verbose=1, tensorboard_log='C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/tensorboard/multi_maskppo')
    
    # model.learn(int(2e5), progress_bar=True)
    
    # model = PPO.load('./model/ppo_model2', env = env, tensorboard_log='./tensorboard/multi_maskppo')
    
    #callback_on_best = StopTrainingOnRewardThreshold(max_no_improvement_evals=100, verbose=1)
    
    # 왜안쓰냐 이거 아래 주석 함
    eval_callback = EvalCallback(eval_env=gym.make(env_id,  ct2_threshold=20, print_Map=False),
                    eval_freq=100,
                    render=False,
                    best_model_save_path = './logs/best_model',
                    log_path = './log/results',
                    verbose=1)
    
    # Checkpoint_Callback = CheckpointCallback(save_freq=1000, save_path='./logs/', verbose=1)
    
    callback = CallbackList([rollout_callback, eval_callback])
    
    model.learn(total_timesteps=int(2e4), callback=callback,
                log_interval=10,
                progress_bar=True)
    
    # Save
    model.save('./model/mask_multi_myppo_v2.zip')