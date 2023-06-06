import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from .base_skill import BaseSkill 

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


def check_go_through(pos, maps):
    x, y = pos 
    width, height, _ = maps.shape
    if x<0 or x>=width or y<0 or y>=height:
        return False
    return (maps[x, y, 0] in [1, 8] or (maps[x, y, 0] == 4 and maps[x, y, 2]==0) )

def get_neighbors(pos_and_dir, maps):
    x, y, direction = pos_and_dir
    next_dir_left = direction - 1 if direction > 0 else 3
    next_dir_right = direction + 1 if direction < 3 else 0
    neighbor_list = [(x,y,next_dir_left), (x,y,next_dir_right)]
    forward_x, forward_y = DIRECTION[direction]
    new_x,new_y = (x+forward_x, y+forward_y)
    
    if check_go_through((new_x,new_y), maps):
        neighbor_list.append((new_x, new_y, direction))

    
    assert not len(neighbor_list)==0
    
    return neighbor_list

    
class GoTo_Goal(BaseSkill):
    def __init__(self, init_obs, target_pos):
        ''' 
            Inputs:
                init_obs: {'image': width x height x 4 , 'mission': ,}
                target_pos: (x,y)
        '''
        obs = init_obs[:,:,-4:] ## why here init_obs is [:,:,0]
        self.unpack_obs(obs)
        self.obs = obs
        x, y = target_pos
        target_pos_and_dir = [(x-1, y, 0), (x, y-1, 1), (x+1, y, 2), (x, y+1, 3)]
        self.path = self.plan(target_pos_and_dir) 
        # print(self.path[0], self.path[-1])
        
    def plan(self, target_pos_and_dir):
        start_node = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
        
        open_list = set([start_node])
        closed_list = set([])
        
        g = {}
        g[start_node] = 0
        
        parents = {}
        parents[start_node] = start_node
        
        while len(open_list) > 0:
            n = None
            
            for v in open_list:
                if n is None or g[v] < g[n]:
                    n = v
                    
            if n == None:
                print('No path found!!')
                return None
            
            ### reconstruct and return the path when the node is the goal position
            if n in target_pos_and_dir:
                reconst_path = []
                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]
                    
                reconst_path.append(start_node)
                reconst_path.reverse()
                return reconst_path
                
            for m in get_neighbors(n, self.obs):
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + 1
                    
                else:
                    if g[m] > g[n]+1:
                        g[m] = g[n]+1
                        parents[m] = n
                        
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            
            open_list.remove(n)
            closed_list.add(n)
            
        print('No path found!!!')
        print(self.obs[:,:,0].T)
        print(self.obs[:,:,3].T)
        print(target_pos_and_dir)
        return []
    
    def step(self, obs=None):
        if obs is not None:
            obs = obs[:,:,-4:]
            self.unpack_obs(obs)
            cur_pos, cur_dir = self.agent_pos, self.agent_dir         
            cur_agent = self.path.pop(0)
            assert cur_pos[0] == cur_agent[0] 
            assert cur_pos[1] == cur_agent[1]
            assert cur_dir == cur_agent[2]
        else:
            cur_dir = self.path.pop(0)[2]
        
        next_dir = self.path[0][2]
        angle = (cur_dir - next_dir) % 4 
        if angle == 1:
            action = 0
        elif angle == 3:
            action = 1
        elif angle == 0:
            action = 2
        else:
            print('No action find error!!!')
            action = None
            
        done = self.done_check()
        return action, done

    def done_check(self):
        return len(self.path)<=1