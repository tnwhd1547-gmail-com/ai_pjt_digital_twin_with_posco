import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np

from .binpacking_posco_v0 import binpacking_posco_v0

class binpacking_posco_v1(binpacking_posco_v0):
    # 22개
    products = [(3,3), (3,3), (3,3), (1,1), (1,1), (1,1),
                (3,3), (3,3), (3,3), (1,1), (1,1), (1,1),
                (2,2), (2,2), (2,2), (2,2), (2,2),
                (2,2), (2,2), (2,2), (2,2), (2,2), (10,10)]
    # 마지막은 종료용 Empty
    """
    version 1
    
    v0 의 모든 함수를 따라감.
    변경부분만 아래에 적어 두었음.
    물리적으로 불가능한 액션 시행시 -1점을 받고 내부적으로 기록하도록.
    """
    def __init__(self, **kwargs):
        super(binpacking_posco_v1, self).__init__(**kwargs)
        self.prod_idx = 0 # 랜덤이 아닐 때 product index
        
        self.width = self.products[self.prod_idx][0]
        self.length = self.products[self.prod_idx][1]
    
    def update_product(self): # get next product
        self.prod_idx += 1
        self.width = self.products[self.prod_idx][0]
        self.length = self.products[self.prod_idx][1]
        self.filled_map += self.width*self.length

    def step(self, action):
        action = self.int_action_to_grid(action)
        
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.filled_map > 80 # 80% 이상
            or self.prod_idx == 22
        )
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                self.update_product()
                self.state = np.append(self.Map.flatten(), self.width)
                reward = 1
                self.ct2 = 0
            else:
                reward = -1
        else:
            reward = -1
        info = {'score' : score}
        
        return self.state, reward, terminated, info
    
    def reset(self):
        self.ct2 = 0
        self.filled_map = 0
        
        self.prod_idx = 0
        self.width = self.products[self.prod_idx][0]
        self.length = self.products[self.prod_idx][1]
        
        # 매번 다른 state를 주고 학습 시킬 수 있음 !
        self.state = np.append(self.Map.flatten(), self.width)
        
        # Map of warehouse
        self.Map = np.zeros(self.mapsize, dtype=int)
        
        return np.array(self.state)