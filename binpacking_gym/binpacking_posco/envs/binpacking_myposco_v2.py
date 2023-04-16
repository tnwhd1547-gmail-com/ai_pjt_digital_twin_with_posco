import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np
from gym import spaces
from .binpacking_posco_v0 import binpacking_posco_v0

class binpacking_myposco_v2(binpacking_posco_v0):
    """
    Version 2
    v1의 모든 함수를 따라감 (Product 순서 정해짐)

    threshold 비율 만큼 물건을 채웠을 때 추가 점수
    -> Masking 적용 안됨.
    """
    
    products_list = [(22, 19), (19, 22), (27, 18), (18, 27),
                (34, 25), (25, 34)]
    
    def __init__(self, **kwargs):
        super(binpacking_myposco_v2, self).__init__(**kwargs)
        self.fill_threshold = kwargs.get('fill_threshold', 0.8)
        self.mapsize = kwargs.get('mapsize', [100, 100])
        # For ending of episode
        self.filled_map = 0 
        self.ct2 = 0
        
        # Product's Size
        self.random_product()
        
        # Map of warehouse
        self.Map = np.zeros(self.mapsize, dtype=int)
        self.max_x = self.mapsize[0]-1
        self.max_y = self.mapsize[1]-1
        
        self.state = None
        self.actions_grid = [[i, j] for j in range (self.max_x+1) for i in range(self.max_y+1)]
        ### 여기부터 있어야 되는 것들
        self.action_space = spaces.Discrete(len(self.actions_grid)+1)
        # Observation space
        ## Map + Max box size
        low = np.array([0 for _ in range(len(self.actions_grid))] + [0, 0]) 
        high = np.array([1 for _ in range(len(self.actions_grid))] + [4, 4])
        self.observation_space = spaces.Box(low, high, dtype=int)
        self.rotate = False
        
    
    def random_product(self):
        product = self.products_list[np.random.choice(len(self.products_list))]
        self.width = product[0]
        self.length = product[1]
        self.filled_map += self.width * self.length
    
    def step(self, action):
        # action = self.int_action_to_grid(action)
        try:
            action[0]
        except:
            action = list(divmod(action, 10))
            
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.filled_map * self.fill_threshold > 10000 * self.fill_threshold
        )
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                # 중간 리워드 줘야할 것 같은데.. 채운 공간만큼 줘야할듯
                reward = self.length * self.width
                # self.update_product()
                self.state = np.append(self.Map.flatten(), [self.width, self.length])
                self.ct2 = 0
            else:
                self.ct2 += 1
                reward = -1
        else:
            if self.filled_map >= 10000*self.fill_threshold:
                reward = self.filled_map/5 # 80 - 100
            else:
                reward = -1
        info = {'score' : score}
        
        return self.state, reward, terminated, info
    
    def reset(self):
        self.ct2 = 0
        self.filled_map = 0
        self.random_product()
        self.rotate=False
        # 매번 다른 state를 주고 학습 시킬 수 있음 !
        # Map of warehouse
        self.Map = np.zeros(self.mapsize, dtype=int)
        self.state = np.append(self.Map.flatten(), [self.width, self.length])

        return np.array(self.state) #self.state # np.array(self.state)
    
    def render(self, action, reward, mode='human'):
        # action2 = action[::-1]
        print (f'Action :{action}, box :{self.width}, {self.length}, reward : {reward}')
        print(self.ct2, self.ct2_threshold, 'map_filled : ', sum(sum(self.Map)))
        self.print_Map = True
        if self.print_Map:
            print (self.Map)
            
    
    def close(self):
        pass
    
    # get_action_mask 추가
    
    def mask_action(self):
        act = [self.available_act(self.actions_grid[i]) for i in range(len(self.actions_grid))]
        if self.rotate:
            act.append(False)
        else:
            act.append(True)
        
        return act