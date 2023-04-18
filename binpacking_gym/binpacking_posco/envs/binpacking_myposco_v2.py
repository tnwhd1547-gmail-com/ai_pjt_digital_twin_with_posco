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
    # 기존에 훈련 시키려고 했던 박스 크기들
    # products_list = [(22, 19), (19, 22), (27, 18), (18, 27),
                # (34, 25), (25, 34)]
    products_list = [(4, 3), (3, 4), (9, 6), (6, 9),
                (12, 13), (13, 12)]
    
    def __init__(self, **kwargs):
        super(binpacking_myposco_v2, self).__init__(**kwargs)
        self.fill_threshold = kwargs.get('fill_threshold', 0.8)
        self.mapsize = kwargs.get('mapsize', [50, 50])
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

    def int_action_to_grid(self, action):
        return (self.actions_grid[action])
    
    def available_act(self, action):
        """
        선택한 Action이 시행 가능한지 확인
        """        
        # print("available_action", action)
        # 요놈도 주석처리 했음 
        # self.ct2 += 1 # count unavailable action
        if action[0] + self.length > self.max_x + 1:
            return False
        if action[1] + self.width > self.max_y + 1:
            return False
        if self.Map[action[0]][action[1]] == 1:
            return False
        # print('map : ', self.Map)
        
        if self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)].sum() > 0:
            return False
        # if self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)].sum() > 0:
        #     return False
        return True
    
    def map_action(self, action):
        """
        물건을 내려놓고 Map을 0 -> 1로 변경
        """        
        # Drop product (Only for Square) / Fill the Map
        # self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)] = 1
        #아래 코드로 변경
        # print('action ', action)
        # print('map', self.Map)
        # print('type', type(action), type(3))
        # if type(action)==type(3):
        #     action = self.int_action_to_grid(action)
        self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)] = 1
    
    def step(self, action):
        # action = self.int_action_to_grid(action)
        
        if action == 2500:
            self.width, self.length = self.length, self.width
            self.rotate = True
            self.state = np.append(self.Map.flatten(), [self.width, self.length])

            return self.state, 0, False, {}
        
        ## 변경함 4 17
        action = self.int_action_to_grid(action)
        # try:
        #     action[0]
        # except:
        #     action = list(divmod(action, 100))
            
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or sum(sum(self.Map)) * self.fill_threshold >= 2500 * self.fill_threshold
        )
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                # 중간 리워드 줘야할 것 같은데.. 채운 공간만큼 줘야할듯
                reward = self.length * self.width
                # self.update_product()
                self.state = np.append(self.Map.flatten(), [self.width, self.length])
                self.random_product()
                self.ct2 = 0
            else:
                self.ct2 += 1
                reward = -(self.length * self.width)/12
        else:
            if sum(sum(self.Map)) >= 2500*0.7:#self.fill_threshold:
                reward = self.filled_map/5 # 80 - 100
            else:
                reward = 0 #- 50
            # 못채웠을때 - 점수 없앰
        info = {'score' : score}
        
        return self.state, reward, terminated, {}
    
    def reset(self):
        self.ct2 = 0
        self.filled_map = 0
        self.random_product()
        self.rotate=False
        # 매번 다른 state를 주고 학습 시킬 수 있음 !
        # Map of warehouse
        self.Map = np.zeros(self.mapsize, dtype=int)
        self.state = np.append(self.Map.flatten(), [self.width, self.length])

        return self.state # np.array(self.state)
    
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