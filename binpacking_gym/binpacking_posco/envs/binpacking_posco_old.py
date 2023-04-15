import numpy as np
import gym
from gym import spaces
from random import choice
from copy import copy
"""
For stable-baseline,
No masking
When agent pick unavailable action, resample with While Loop.
"""

class binpacking_posco_v0(gym.Env):
    """
    Custom Env for 2D-bin Packing
    """
    # Class Variables 
    products = [(4, 4), (2, 2), (3, 3), (1, 1), (2, 2), (3, 3), (4, 4), (1, 1), (2, 2), (1, 1),
                (4, 4), (2, 2), (3, 3), (1, 1), (2, 2), (3, 3), (4, 4), (1, 1), (2, 2), (1, 1)]
    
    def __init__(self, first = True):
        if first: # for reset
            super().__init__()

        # Product's Size
        # 여기서는 애초에 ㄱ자, ㄴ자 정의가 안됨
        self.ct = 0 # index of products
        self.width = self.products[self.ct][0]
        self.length = self.products[self.ct][1]
        
        # Map of warehouse
        self.Map = np.zeros([10, 10])
        self.max_x = self.Map.shape[0]-1
        self.max_y = self.Map.shape[1]-1
        
        self.actions_grid = [[i, j] for j in range (self.max_x+1) for i in range(self.max_y+1)]
        #self.action_space = [i for i in range(len(self.actions_grid))] # Action's Index
        self.action_space = spaces.Discrete(len(self.actions_grid))
        self.observation_space = spaces.MultiBinary([self.max_x+1, self.max_y+1])
    
    def update_product(self, ct):
        self.width = self.products[ct][0]
        self.length = self.products[ct][1]

    def available_act(self, done):
        """
        Get custom action_space
        Select available grid by product's size,
        then return available action_state. -> index(actions)
        
        If no available space = Done (End episode.)
        """
        available_act = []
        for i in range(len(self.actions)):
            # Max x, y
            if self.actions[i][0] + self.width > self.max_x:
                continue
            if self.actions[i][1] + self.length > self.max_x:
                continue
            
            # 직사각형은 되고 ㄱ 자 는 안됨
            if sum(self.Map[self.actions[i][0]:self.actions[i][0]+self.width][self.actions[i][1]:self.actions[i][1]+self.length]) != 0:
                available_act.append(i)
            else:
                continue
        
        if available_act == []:
            done = True
        return available_act

    def map_action(self, done):
        """
        Randomly sample ACT with available act list.
        
        Actions of Map
        1. Change Map (Drop product)
        2. Change action_space
        """
        # End episode
        if done == True:
            return ()
        
        # Sampled ACT
        act = choice(self.available_act(done=done))
        
        # Drop product
        # Only for Square
        self.Map[self.actions[act][0]:self.actions[act][0]+self.width][self.actions[act][1]:self.actions[act][1]+self.length] = 1
        
        for i in range(self.actions[act][0], self.actions[act][0]+self.width):
            for j in range(self.actions[act][1], self.actions[act][1]+self.length):
                self.actions.remove([i, j]) # remove actions

    def step(self, action):
        self.map_action(action)

        self.ct += 1
        self.update_product(self.ct)

        obs = self.Map
        done = False
        if done:
            reward = 0#reward function
        else:
            reward = 0       
        
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.__init__(False)
        # self.ct = 0 # index of products
        # self.update_product(self.ct)
        # self.Map = np.zeros([10, 10])
        # self.action_space = spaces.Discrete(5) # number of actions
        
        return (self.Map)
    
    def render(self, mode):
        pass
    


#=================
def main():
    import gym
    import binpacking_posco
    env = gym.make('binpacking_posco-v0')
    env = gym.make('CartPole-v1')
    
    env.reset()
    env.step(1)
    
    for episode in range(10):
        env.reset() # first
        done = False
        while not done:
            action = env.actions[env.action_space.sample()]
            new_state, reward, done, info = env.step(action)
    
    env.close()
    
    from stable_baselines3.common.env_checker import check_env
    check_env(env)
    
    from sb3_contrib.common.maskable.utils import get_action_masks
    get_action_masks(env)
