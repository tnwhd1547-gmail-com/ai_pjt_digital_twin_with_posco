import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np
from gym import spaces
from .binpacking_poscopredict_v00 import binpacking_poscopredict_v00

class binpacking_poscopred_v1(binpacking_poscopredict_v00):
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
        super(binpacking_poscopred_v1, self).__init__(**kwargs)
        ### 여기부터 있어야 되는 것들
        self.action_space = spaces.Discrete(len(self.actions_grid)+1)
        # Observation space
        ## Map + Max box size
        low = np.array([0 for _ in range(len(self.actions_grid))] + [0, 0]) 
        high = np.array([1 for _ in range(len(self.actions_grid))] + [4, 4])
        self.observation_space = spaces.Box(low, high, dtype=int)
        self.rotate = False
        
        self.prod_idx = 0 # 랜덤이 아닐 때 product index
        
        self.width = self.products[self.prod_idx][0]
        self.length = self.products[self.prod_idx][1]
        self.ct2_threshold = kwargs.get('ct2_threshold', 40)
    
    def update_product(self): # get next product
        self.prod_idx += 1
        self.width = self.products[self.prod_idx][0]
        self.length = self.products[self.prod_idx][1]
        self.filled_map += self.width*self.length

    def step(self, action):
        
        if action == 100:
            self.width, self.length = self.length, self.width
            self.rotate = True
            self.state = np.append(self.Map.flatten(), [self.width, self.length])
            info = {'box' :[self.width, self.length],'box_filled':sum(sum(self.Map))}
            return self.state, 0, False, info
        
        action = self.int_action_to_grid(action)
        
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.filled_map >= 80 # 80% 이상
            or self.prod_idx == 22
        )
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                self.update_product()
                self.state = np.append(self.Map.flatten(), [self.width,self.length])
                reward = 1
                self.ct2 = 0
            else:
                self.ct2 += 1
                reward = -1
        else:
            reward = -1
        # info = {'score' : score}
        info = {'box' :[self.width, self.length],'box_filled':sum(sum(self.Map))}
        
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
        
        return self.state #np.array(self.state)