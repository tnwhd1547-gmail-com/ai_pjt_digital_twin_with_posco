import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np

from binpacking_posco_v1 import binpacking_posco_v1

class binpacking_posco_v2(binpacking_posco_v1):
    """
    Action Masking version
    """
    def valid_action_mask(self):
        """
        현재 상태에서 가능한 Action ndarray
        """
        mask_action = []
        for i in range(len(self.actions_grid)):
            if self.available_act(self.actions_grid[i]):
                mask_action.append(i)
            self.ct2 -= 1
        
        return np.array(mask_action)