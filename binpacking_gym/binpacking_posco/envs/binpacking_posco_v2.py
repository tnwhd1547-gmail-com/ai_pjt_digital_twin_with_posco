import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np

from .binpacking_posco_v1 import binpacking_posco_v1

class binpacking_posco_v2(binpacking_posco_v1):
    """
    Version 2
    v1의 모든 함수를 따라감 (Product 순서 정해짐)

    threshold 비율 만큼 물건을 채웠을 때 추가 점수
    -> Masking 적용 안됨.
    """
    
    def __init__(self, **kwargs):
        super(binpacking_posco_v2, self).__init__(**kwargs)
        self.fill_threshold = kwargs.get('fill_threshold', 0.8)
    
    def step(self, action):
        action = self.int_action_to_grid(action)
        
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.prod_idx == 22
        )
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                # 중간 리워드 줘야할 것 같은데.. 채운 공간만큼 줘야할듯
                reward = 1
                self.update_product()
                self.state = np.append(self.Map.flatten(), self.width)
                self.ct2 = 0
            else:
                reward = -1
        else:
            if self.filled_map >= 100*self.fill_threshold:
                reward = self.filled_map/5 # 80 - 100
            else:
                reward = -1
        info = {'score' : score}
        
        return self.state, reward, terminated, info
