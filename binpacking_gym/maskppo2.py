import gym
import numpy as np
import stable_baselines3
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import binpacking_posco
from stable_baselines3.common.callbacks import EvalCallback
from acm import mask_fn
from stable_baselines3.common.evaluation import evaluate_policy

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


###
# class EvalCallback(stable_baselines3.common.callbacks.BaseCallback):
#     def __init__(self, eval_env, callback_on_new_best=None, eval_freq=10000, n_eval_episodes=5,
#                  log_path=None, best_model_save_path=None, deterministic=True, render=False, verbose=0):
#         super().__init__(verbose)
#         self.eval_env = eval_env
#         self.callback_on_new_best = callback_on_new_best
#         self.eval_freq = eval_freq
#         self.n_eval_episodes = n_eval_episodes
#         self.log_path = log_path
#         self.best_model_save_path = best_model_save_path
#         self.deterministic = deterministic
#         self.render = render
#         self.evaluations = {'eval/ep_rew_mean': [], 'eval/ep_rew_std': [], 'eval/success_rate': []}
#         self.best_mean_reward = -np.inf
    
#     def _on_step(self) -> bool:
#         return True

#     def on_step_end(self) -> bool:
#         if self.n_calls % self.eval_freq == 0:
#             eval_env = self.eval_env
#             mean_reward, std_reward, success_rate = evaluate_policy(self.model, eval_env,
#                                                                      n_eval_episodes=self.n_eval_episodes,
#                                                                      deterministic=self.deterministic,
#                                                                      render=self.render)
#             self.evaluations['eval/ep_rew_mean'].append(mean_reward)
#             self.evaluations['eval/ep_rew_std'].append(std_reward)
#             self.evaluations['eval/success_rate'].append(success_rate)

#             if self.callback_on_new_best is not None:
#                 if mean_reward > self.best_mean_reward:
#                     self.best_mean_reward = mean_reward
#                     if self.best_model_save_path is not None:
#                         self.model.save(self.best_model_save_path)

#                     if self.log_path is not None:
#                         with open(self.log_path, 'a') as f:
#                             f.write(f"n_calls: {self.n_calls}, best_mean_reward: {self.best_mean_reward}\n")

#                     self.callback_on_new_best(mean_reward, self.model)
#         return True

###
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
                    deterministic = True,
                    callback_on_new_best=None,
                    verbose=1)
# model = MaskablePPO.load('./model/mask_multi_myppo_v2_ver1', env=env, tensorboard_log='./tensorboard/my_maskppo')

# eval_callback.evaluations = {'eval/ep_rew_mean': [], 'eval/ep_rew_std': [], 'eval/success_rate': []}

model.learn(int(10e3), callback=eval_callback, progress_bar=True)
# eval_callback.on_training_start(locals(), globals())
# eval_ep_rew_std = eval_callback.evaluations["eval/ep_rew_std"][-1]
# eval_success_rate = eval_callback.evaluations["eval/success_rate"][-1]
# f = open("eval_ep_rew_std", 'w')
# for i in range(eval_ep_rew_std):
#     f.write(f'{i}\n')
# f.close()

# f2 = open("eval_success_rate", 'w')
# for i in range(eval_success_rate):
#     f2.write(f'{i}\n')
# f2.close()

model.save('./model/my_maskppo_mynewv2_thres7')
# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
# model.predict(observation, action_masks=valid_action_array)