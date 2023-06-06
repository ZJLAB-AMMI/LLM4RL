import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from .base_skill import BaseSkill 

from utils import global_param
class Explore(BaseSkill):
    def __init__(self, init_obs, view_size, exp=None):
        ''' 
            Inputs:
                init_obs: {'image': width x height x channel, 'position': [x,y], 'direction': , 'mission': ,}
                view_size: env.agent_view_size
        '''
        
        self.view_size = view_size
        self.scope = self.view_size // 2
        self.reset_to_NW = exp.reset_to_NW if exp is not None else False
        init_obs = init_obs[:,:,-4:]
        self.path = exp.path if exp is not None else self.plan(init_obs)
        self.unpack_obs(init_obs) 
        # self.reset_to_NW = global_param.get_value('reset_to_NW')

        # self.path = global_param.get_value('path')
        #self.path = self.plan(init_obs) 
        #self.reset_to_NW = False 

    def plan(self, init_obs):
        self.unpack_obs(init_obs) 
        if self.agent_dir == 0: # turn until facing west
            path = [0, 0]
        elif self.agent_dir == 1:
            path = [1]
        elif self.agent_dir == 2:
            path = []
        else:
            path = [0]
        return path
        
    def get_fwd_obj(self):
        if self.agent_dir == 0:
            fwd_obj = self.map[self.agent_pos[0] + self.scope, self.agent_pos[1]]
        elif self.agent_dir == 2:
            fwd_obj = self.map[max(0, self.agent_pos[0] - self.scope), self.agent_pos[1]]
        elif self.agent_dir == 1:
            fwd_obj = self.map[self.agent_pos[0], self.agent_pos[1] + self.scope]
        else:
            fwd_obj = self.map[self.agent_pos[0], max(0, self.agent_pos[1] - self.scope)]
        return fwd_obj
    
    def step(self, obs):
        obs = obs[:,:,-4:]
        self.unpack_obs(obs)
        fwd_obj = self.get_fwd_obj()
       
        if self.reset_to_NW == False:
            if len(self.path) > 0:
                action = self.path.pop(0)
               
            elif fwd_obj == 2 or fwd_obj == 4: # hitting a wall or hitting a door
                action = 1 # west -> north -> east
                if self.agent_dir == 3: # facing north
                    self.reset_to_NW = True
            else:
                action = 2
        else:
            if fwd_obj == 2 or fwd_obj == 4: # hitting a wall
                if self.agent_dir == 0: # facing east
                    self.path = [2] * self.view_size + [1] # go south for self.view_size step, then turn west
                    action = 1 # east -> south
                elif self.agent_dir == 2: # facing west
                    self.path = [2] * self.view_size + [0] # go south for self.view_size step, then turn east
                    action = 0 # west -> south
                elif self.agent_dir == 1: # facing south
                    if self.path == []:
                        action = np.random.choice([0,1]) 
                    else:
                        action = self.path[-1]
                    self.path = []
                else:
                    action = 1
                    self.path = []
            elif len(self.path) > 0:
                action = self.path.pop(0)
            else:
                action = 2     
        done = self.done_check()    
        if done:
            global_param.set_value('explore_done', True)

        return action, done
    
    def done_check(self):
        done = False
        ## Position the boundary, because the room is rectangular
        wall = np.argwhere(self.map == 2) 
        door = np.argwhere(self.map == 4)
        boundary = np.concatenate((wall, door))
        ## if exploer over, wall must is loop in grid
        if FindLoop(boundary):
            lower_left = boundary.min(axis=0)
            upper_right = boundary.max(axis=0)
            room = self.map[lower_left[0]:upper_right[0]+1,lower_left[1]:upper_right[1]+1]
            done = np.count_nonzero(room == 0) == 0 # no unseen
        return done


# DFS way to find loop
def FindLoop(boundary):
    if boundary.size == 0:
        return False
    max_x,max_y = boundary.max(axis=0)
    boundary = boundary.tolist()

    visited = set()
    def dfs(x,y, parent):
        # return True if find loop
        if (x,y) in visited:
            return True
        visited.add((x,y))
        for direct in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            new_x = x + direct[0]
            new_y = y + direct[1]
            if new_x < 0 or new_x > max_x or new_y < 0 or new_y > max_y:
                continue
            if (new_x, new_y) != parent and [new_x,new_y] in boundary:
                if dfs(new_x, new_y, (x,y)):
                    return True
        return False    
    x0 = boundary[0][0]
    y0 = boundary[0][1]
    loop = dfs(x0, y0, None)
    return loop


if __name__ == "__main__":
    import numpy as np
    visited = set()
    loop = False
    boundary = np.array([[0,0],[0,1],[0,2],[1,0],[1,2],[2,2],[2,1],[2,0]])
    print(FindLoop(boundary))


