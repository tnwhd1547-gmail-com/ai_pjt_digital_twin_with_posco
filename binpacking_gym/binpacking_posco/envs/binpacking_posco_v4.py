import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np

from .binpacking_posco_v0 import binpacking_posco_v0
from gym import spaces

class binpacking_posco_v4(binpacking_posco_v0):
    """
    - 직사각형 적용 (1 - 4)
    - 회전 가능
    - 물건 랜덤
    - Maskable PPO
    - 랜덤한 물건에대한 최대 reward 목적
    """
    
    def __init__(self, **kwargs):
        super(binpacking_posco_v4, self).__init__(**kwargs)
        low = np.array([0 for _ in range(len(self.actions_grid))] + [0, 0]) 
        high = np.array([1 for _ in range(len(self.actions_grid))] + [4, 4])
        self.action_space = spaces.Discrete(len(self.actions_grid)+1) # 회전추가
        self.observation_space = spaces.Box(low, high, dtype=int)
        self.rotate = False
    
    def random_product(self):
        self.width = np.random.choice(range(1,4))
        self.length = np.random.choice(range(1,4))
        self.filled_map += self.width*self.length
        
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
        act = [self.available_act(self.actions_grid[i]) for i in range(len(self.actions_grid))]
        if self.rotate:
            act.append(False)
        else:
            act.append(True)
        
        return act
    
    def step(self, action):
        if action == 100:
            self.width, self.length = self.length, self.width
            self.rotate = True
            self.state = np.append(self.Map.flatten(), [self.width, self.length])

            return self.state, 0, False, {}
            
        action = self.int_action_to_grid(action)
        
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.filled_map >= 100
            #or self.valid_action_mask.sum() == 0
        )
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                reward = self.width * self.length 
                self.state = np.append(self.Map.flatten(), [self.width, self.length])
                self.random_product()
                self.ct2 = 0
            else:
                self.ct2 += 1
                reward = 0
        else:
            reward = 0
        
        return self.state, reward, terminated, {}

    def reset(self):
        # Reset의 Return이 이전과 많이 다름 주의!
        self.ct2 = 0
        self.filled_map = 0
        self.random_product()
        self.rotate=False
        self.Map = np.zeros(self.mapsize, dtype=int)
        self.state = np.append(self.Map.flatten(), [self.width, self.length])

        return self.state