import gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
# 환경 생성
env_id = 'CartPole-v1'
env = gym.make(env_id)
env = Monitor(env, './logs')
env = DummyVecEnv([lambda: env])
# ActionMasker 래퍼 적용
action_mask_fn = lambda obs, mask: obs * mask
env = ActionMasker(env, action_mask_fn=action_mask_fn)
# 모델 생성 및 학습
model = PPO(policy="MlpPolicy", env=env, verbose=1)#, action_mask=[1, 1, 0, 0])
model.learn(total_timesteps=10000)

# 모델 저장
model.save("mask_ppo_model")

# 액션 마스킹 적용하여 테스트
obs = env.reset()
while True:
    # 액션 마스킹 적용
    action, _states = model.predict(obs, mask=[1, 1, 0, 0])
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

# 환경 종료
env.close()
