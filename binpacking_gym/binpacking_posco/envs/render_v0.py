#-*- coding:utf-8 -*-
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import gym
from gym import spaces
from random import choice
from copy import copy

"""
가장 간단한 환경
Product : Random
"""

class binpacking_posco_v0(gym.Env):
    """
    Custom Env for 2D-bin Packing
    """
    # Class Variables 
    products_list = [(1,1), (2,2), (3,3)] # for random sampling
    
    # reward_range = (0, 100)
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps" : 4}
    spec = "EnvSpec"
    
    
    def __init__(self, render_mode = "human", size = 10, **kwargs):
        """
        Kwargs / Default value (type)
        -------------
        ct2_threshold : 불가능한 행동을 몇번까지 허용할 것인가 / 50 (int)
        # threshold 몇 %의 맵을 채웠을때 추가점수를 줄 것인가. : / 0.6 (float)
        mapsize : 전체 맵 사이즈 / [10, 10] (list)
        print_Map : Action시 마다 Map 출력 / True (bool)
        """
        super(binpacking_posco_v0, self).__init__()
        # Params
        self.ct2_threshold = kwargs.get('ct2_threshold', 20) # 불가능한 행동의 제한 수
        self.mapsize = kwargs.get('mapsize', [10, 10])
        self.print_Map = kwargs.get('print_Map', True)
        self.threshold = kwargs.get('threshold', 0.6) # Default = 0.6 # 이 비율의 공간을 채웠을 때 더 많은 리워드를 줌
        
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
        self.action_space = spaces.Discrete(len(self.actions_grid)) # Discete(100)

        # Observation space
        ## Map + Max box size
        low = np.array([0 for _ in range(len(self.actions_grid))] + [0])
        high = np.array([1 for _ in range(len(self.actions_grid))] + [4])
        self.observation_space = spaces.Box(low, high, dtype=int)

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.size = size
        self.window_size = 400

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
        Maskable PPO 돌릴 때는 self.ct2 카운트 올리는 부분 주석처리할 것
        """        
        # self.ct2 += 1 # count unavailable action
        if action[0] + self.length > self.max_x + 1:
            return False
        if action[1] + self.width > self.max_y + 1:
            return False
        if self.Map[action[0]][action[1]] == 1:
            return False
        
        if self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)].sum() > 0:
            return False
        
        return True

    def map_action(self, action):
        """
        물건을 내려놓고 Map을 0 -> 1로 변경
        """        
        # Drop product (Only for Square) / Fill the Map
        self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)] = 1

    def step(self, action):
        action = self.int_action_to_grid(action)
        self.render_action = action

        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.filled_map > 100 # 100% 이상으로 변경
        )
        
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                self.random_product()
                self.state = np.append(self.Map.flatten(), self.width)
                reward = self.width * self.length
                self.ct2 = 0
            else:
                reward = 0
        else:
            reward = 0
        
        info = {'score' : score}
        
        # Rendering
        if self.render_mode == "human":
            self._render_frame()
        
        return self.state, reward, terminated, info

    def reset(self):
        self.ct2 = 0
        self.filled_map = 0
        
        self.random_product()
        
        # 매번 다른 state를 주고 학습 시킬 수 있음 !
        self.state = np.append(self.Map.flatten(), self.width)
        
        # Map of warehouse
        self.Map = np.zeros(self.mapsize, dtype=int)

        # Rendering
        if self.render_mode == "human":
            self._render_frame()
        
        return np.array(self.state)

    # HCAK : mode 변경 없으므로 아래 추후에 바꿀 것
    def render(self, mode='human'):
        if self.render_mode == "human": # default : rgb_array
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":            
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Canvas Initialize
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))
        for x in range(0, 400, 40):
            pygame.draw.line(canvas, (211, 211, 211, 50), (x, 0), (x, 400))
        for y in range(0, 400, 40):
            pygame.draw.line(canvas, (211, 211, 211, 50), (0, y), (400, y))

        # Plot Boxes
        Map = list(map(list, self.Map))
        boxes = []
        for i in range(10) :
            for j in range(10):
                if Map[i][j] == 1:
                    boxes.append((i, j, 1, 1)) # 여기 지금 1,1을 박스 크기별로 바꿔야 함

        for (x, y, w, h) in boxes:
                    if w == 1: boxcolor = (255,204,255) # Pink
                    elif w == 2 : boxcolor = (255,255,102) # Green
                    elif w == 3 : boxcolor = (204,255,255) # Blue
                    pygame.draw.rect(canvas, boxcolor, [x*40, y*40, w*40, h*40])
                    # pygame.time.delay(100) # delay주면 한 episode가 너무 오래걸림

        font = pygame.font.SysFont("arial", 30, True, False)
        text = font.render(f"score : {np.sum(self.Map)}", True, (255,0,0))
        canvas.blit(text, (130,200))
        pygame.display.update()
        pygame.display.flip()

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes = (1,0,2)
            )
            
        pygame.display.set_caption("Rendering")

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class binpacking_posco_v1(binpacking_posco_v0):
    pass
