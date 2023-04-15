import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np

from .binpacking_posco_v1 import binpacking_posco_v1

class binpacking_posco_v3(binpacking_posco_v1):
    """
    version 1을 따라감 (물건 순서 고정)
    Maskable PPO 적용
    
    - Reward

    개별 리워드 지급
    > 물건 적재 시 둔 물건의 크기만큼 즉시 리워드 적용
    > 100%를 꼭 다 못채워도 리워드가 높게 나올텐데. -> 우선 하루 돌려보고 생각해보기.

    - Model

    model v2 사용하지 않고 처음부터 학습시작
    """
    
    def __init__(self, **kwargs):
        super(binpacking_posco_v3, self).__init__(**kwargs)
        #self.fill_threshold = kwargs.get('fill_threshold', 0.8)
        
    def available_act(self, action):
        """
        선택한 Action이 시행 가능한지 확인
        """        
        if action[0] + self.length > self.max_x + 1:
            return False
        if action[1] + self.width > self.max_y + 1:
            return False
        if self.Map[action[0]][action[1]] == 1:
            return False
        
        if self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)].sum() > 0:
            return False
        
        return True
    
    def mask_action(self):
        return [self.available_act(self.actions_grid[i]) for i in range(len(self.actions_grid))]
    
    def step(self, action):
        #print (action)
        #print (type(action))
        action = self.int_action_to_grid(action)
        #print (self.available_act(action))
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.prod_idx == 22 # 22번 물건까지 내려두면 100% 채우게 됨.
            #or self.valid_action_mask.sum() == 0
        )
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                reward = self.width * self.length 
                self.update_product()
                self.state = np.append(self.Map.flatten(), self.width)
                self.ct2 = 0
            else:
                #print ("A")
                self.ct2 += 1
                reward = 0
        else:
            reward = 0

        info = {'score' : score}
        
        return self.state, reward, terminated, info
