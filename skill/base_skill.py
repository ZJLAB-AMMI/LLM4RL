import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from minigrid.core.constants import DIR_TO_VEC

'''
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}
'''

DIRECTION = {
    0: [1, 0],
    1: [0, 1],
    2: [-1, 0],
    3: [0, -1],
}

class BaseSkill():
    def __init__(self):
        pass
        
    def unpack_obs(self, obs):
        agent_map = obs[:, :, 3]
        self.agent_pos = np.argwhere(agent_map != 4)[0]
        self.agent_dir = obs[self.agent_pos[0], self.agent_pos[1], 3]
        self.map = obs[:, :, 0]
        # print(self.agent_pos, self.agent_dir)
    def step(self, obs=None):
        raise NotImplementedError

    def done_check(self):
        return False
    
# class Pickup(BaseSkill):
#     def __init__(self, init_obs):
#         init_obs = init_obs[:,:,-4:]
#         self.plan(init_obs)
    
#     def plan(self, init_obs):
#         self.unpack_obs(init_obs)
#         if self.map[self.agent_pos[0], self.agent_pos[1]] == 1: #not carrying
#             self.path = [3]
#         else:
#             if self.get_surrounding_obj(2) == 1 and self.get_surrounding_obj(2, 2) != 4: # back
#                 self.path = [0, 0, 4, 0, 0, 3]
#             elif self.get_surrounding_obj(1) == 1 and self.get_surrounding_obj(1, 2) != 4: # right
#                 self.path = [1, 4, 0, 3]
#             elif self.get_surrounding_obj(3) == 1: # left
#                 self.path = [0, 4, 1, 3]
#             else:
#                 print('No path found!!!')
#                 self.path = None
                
#     def get_surrounding_obj(self, angle, distance=1):
#         target_dir = (self.agent_dir + angle) % 4
#         target_pos = self.agent_pos + DIR_TO_VEC[target_dir] * distance
#         target_obj = self.map[target_pos[0], target_pos[1]]
#         return target_obj
    
#     def step(self, obs):
#         action = self.path.pop(0)
#         done = len(self.path) == 0
#         return action, done
        
        
class Pickup(BaseSkill):
    def __init__(self, init_obs):
        init_obs = init_obs[:,:,-4:]
        self.path_prefix = []
        self.path_suffix = []
        self.plan(init_obs)
    
    def plan(self, init_obs, max_tries=30):
        self.unpack_obs(init_obs)
        
        if self.map[self.agent_pos[0], self.agent_pos[1]] == 1: #not carrying
            self.path = [3]
        else:
            angle_list = [0, 2, 1, 3]
            angle_list.remove(0)
            goto_angle = None
            finish = False
            tries = 0
            while not finish:
                search_angle = angle_list.pop(0)
                _drop, _goto = self.can_drop(search_angle)
                tries += 1
                if _drop:
                    self.update_path(search_angle)
                    self.path = self.path2action(self.path_prefix) + [4] + self.path2action(self.path_suffix) + [3]
                    finish = True
                else:
                    # since there is only 1 door, there is at most 1 angle can go to but cannot drop
                    if _goto: 
                        goto_angle = search_angle
                        
                    if len(angle_list) == 0:
                        if goto_angle or tries < max_tries:
                            self.update_path(goto_angle, forward=True)
                            self.agent_dir = (self.agent_dir + goto_angle) % 4
                            self.agent_pos = self.agent_pos + DIR_TO_VEC[self.agent_dir]
                            angle_list = [0, 2, 1, 3]
                            angle_list.remove(2) # not search backward
                            goto_angle = None
                        else:
                            finish = True
                            self.path = []
                            print("path not found!")
                            
    def can_drop(self, angle, distance=1):
        target_dir = (self.agent_dir + angle) % 4
        target_pos = self.agent_pos + DIR_TO_VEC[target_dir] * distance
        target_obj = self.map[target_pos[0], target_pos[1]]
        if target_obj != 1: # not empty
            _drop, _goto = False, False
        else:
            _drop, _goto = True, True
            for i in range(4):
                nearby_pos = target_pos + DIR_TO_VEC[i]
                if self.map[nearby_pos[0], nearby_pos[1]] == 4: # near a door
                    _drop = False
        return _drop, _goto
                            
    def update_path(self, angle, forward=False):
        if forward:
            if angle == 2:
                self.path_prefix += [2, 'f']
                self.path_suffix = [2, 'f'] + self.path_suffix
            elif angle == 1:
                self.path_prefix += [1, 'f']
                self.path_suffix = [2, 'f', 1] + self.path_suffix
            elif angle == 3:
                self.path_prefix += [3, 'f']
                self.path_suffix = [2, 'f', 3] + self.path_suffix
            else:
                self.path_prefix += ['f']
                self.path_suffix = [2, 'f', 2] + self.path_suffix
        else:
            if angle == 2:
                self.path_prefix += [2]
                self.path_suffix = [2] + self.path_suffix
            elif angle == 1:
                self.path_prefix += [1]
                self.path_suffix = [3] + self.path_suffix
            elif angle == 3:
                self.path_prefix += [3]
                self.path_suffix = [1] + self.path_suffix
            else:
                pass
            
    def path2action(self, path):
        angle = 0
        action_list = []
        path.append('f')
        for i in path:
            if i == 'f':
                angle = angle % 4
                if angle == 1:
                    action_list.append(1)
                elif angle == 3:
                    action_list.append(0)
                elif angle == 2:
                    action_list.extend([0, 0])
                else:
                    pass
                angle = 0
                action_list.append(2)
            else:
                angle += i
        return action_list[:-1]
    
    def step(self, obs):
        action = self.path.pop(0)
        done = len(self.path) == 0
        return action, done

    
class Toggle(BaseSkill):
    def __init__(self):
        pass
    
    def step(self, obs):
        return 5, True
    