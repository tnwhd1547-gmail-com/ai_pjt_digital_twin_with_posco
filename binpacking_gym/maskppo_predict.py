import gym
import numpy as np
from collections import deque
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

env = gym.make('binpacking_poscopred-v1', print_Map=False)  # Initialize env
env = ActionMasker(env, get_action_mask)

model = MaskablePPO.load('./model/mask_multi_ppo_model_10e6', env=env, tensorboard_log='./tensorboard/my_maskppo')

# 모델 결과 Rendering
# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render(reward=rewards)


thres = 50
EPISODES = 1000
end = 0
max_fill = 0
for e in range(EPISODES):
        done = False
        score = 0
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        scores, episodes = [], []
        memory = {'action':[], 'box':[]}
        # env 초기화
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        if end==1:
            break
        while not done:
            action = env.action_space.sample()
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            # next_state = np.reshape(next_state, [1, state_size])
            
            # if agent.render:
                # env.render()
            env.render(action, reward=reward)

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            if not memory['box']:
                memory['box'].append(info['box'])
                
            if reward==1:
                action2 = list(divmod(action, 10))
                memory['action'].append([action2[1],action2[0]])
                memory['box'].append(info['box'])
                
            # # 매 타임스텝마다 학습
            # if len(memory) >= model.train_start: # train_start 이후부터 학습 시작
            #     model.train_model()

            score += reward
            # state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                # model.update_target_model()
                score += env.ct2_threshold
                scores.append(score)
                episodes.append(e)
                print(e)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./save_graph/cartpole_dqn.png")
                # plt.plot(episodes, scores, 'b')
                # plt.savefig('savefig_default.png')
                max_fill = max(max_fill, info['box_filled'])
                if info['box_filled']>=80:
                    memory['box']
                    print("episode:", e,"  score:", score, 'box_count : ',len(memory['action']) ,"box_filled : ",info['box_filled'] ,"  memory action:",
                        memory['action'], "box :", memory['box'][:-1])
                    end = 1
print('max_fill : ',max_fill)         