"""
torch, Keras등 일반 모델 러닝을 위해
Env 돌리기
"""
import sys
sys.path.append('C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym')
import gym
import binpacking_posco
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.dqn import MultiInputPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('binpacking_poscoppo-v1',print_Map=True, ct2_threshold=20)
check_env(env)

#model = PPO('MlpPolicy', env, verbose=1)
model = PPO.load('./ppo_model3', env=env, tensorboard_log='C:/workspace/pba/r2dm/ai_pjt_digital_twin_with_posco/binpacking_gym/tensorboard/ppo1')
model.learn(total_timesteps = int(2e7), progress_bar=True)
model.save('./model/ppo_model3')
#del model # remove to demonstrate saving and loading
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)

# # Check Trained Env
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     print (rewards)
#     vec_env.render()