import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from skill import GoTo_Goal, Explore, Pickup, Toggle
from utils import global_param

class Executive_net():
    def __init__(self, skill_list , init_obs=None, agent_view_size=None):
        
        assert len(skill_list) > 0
        
        self.skill_list = skill_list
        self.num_of_skills = len(skill_list)
        self.current_index = -1
        self.agent_view_size = agent_view_size

        self.actor = self.switch_to_next_skill(init_obs)
        self.skill_done = False

        # current_skill = skill_list[0]
        # self.actor, self.empty_actor = self.switch_skill(current_skill, init_obs)
        # self.skill_done = False

    def switch_skill(self, skill, obs):
        self.action = skill['action']
        if self.action == 0:
            exp = global_param.get_value('exp')
            actor = Explore(obs, self.agent_view_size, exp)
            global_param.set_value('exp', actor)
        elif self.action == 1:
            actor = GoTo_Goal(obs, skill['coordinate'])
            global_param.set_value('exp', None)
        elif self.action == 2:
            actor = Pickup(obs)
        elif self.action == 4:
            actor = Toggle()
        else:
            actor = None
        if actor is None or actor.done_check():
            return None
        return actor #, actor.done_check()
    
    def switch_to_next_skill(self, obs):
        # not_valid_actor = True
        actor = None
        while actor is None:
            self.current_index += 1
            if self.current_index >= self.num_of_skills:
                return None # return None when no skill left in list.
            next_skill = self.skill_list[self.current_index]
            actor = self.switch_skill(next_skill, obs)
        return actor
    
    def __call__(self, obs):
        
        if self.actor is None:
            return np.array([6]), True
        
        if self.skill_done:
            self.actor = self.switch_to_next_skill(obs)
            self.skill_done = False
            
        # obs = obs if self.action == 0 else None
        if self.actor is None:
            return np.array([6]), True
        
        action, done = self.actor.step(obs)
        
        if done and self.current_index == self.num_of_skills - 1:
            return action, True
        elif done:
            self.skill_done = True
            return action, False
        else:
            return action, False


# OBJECT_TO_IDX = {
#     "unseen": 0,
#     "empty": 1,
#     "wall": 2,
#     "floor": 3,
#     "door": 4,
#     "key": 5,
#     "ball": 6,
#     "box": 7,
#     "goal": 8,
#     "lava": 9,
#     "agent": 10,
# }

# STATE_TO_IDX = {
#     "open": 0,
#     "closed": 1,
#     "locked": 2,
# }

# DIRECTION = {
#     0: [1, 0],
#     1: [0, 1],
#     2: [-1, 0],
#     3: [0, -1],
# }



# def check_go_through(pos, maps):
#     x, y = pos 
#     width, height, _ = maps.shape
#     if x<0 or x>=width or y<0 or y>=height:
#         return False
#     return (maps[x, y, 0] in [1, 8] or (maps[x, y, 0] == 4 and maps[x, y, 2]==0) )

# def get_neighbors(pos_and_dir, maps):
#     x, y, direction = pos_and_dir
#     next_dir_left = direction - 1 if direction > 0 else 3
#     next_dir_right = direction + 1 if direction < 3 else 0
#     neighbor_list = [(x,y,next_dir_left), (x,y,next_dir_right)]
#     forward_x, forward_y = DIRECTION[direction]
#     new_x,new_y = (x+forward_x, y+forward_y)
    
#     if check_go_through((new_x,new_y), maps):
#         neighbor_list.append((new_x, new_y, direction))

    
#     assert not len(neighbor_list)==0
    
#     return neighbor_list
    
# class GoTo_Goal():
#     def __init__(self, obs, target_pos):
#         ''' 
#             Inputs:
#                 obs: {'image': width x height x channel, 'position': [x,y], 'direction': , 'mission': ,}
#                 target_pos: (x,y)
#         '''

#         self.obs = obs['image']
#         self.goal = target_pos
#         self.width, self.height, _ = self.obs.shape
#         self.map = self.obs[:,:,0]
#         self.agent_pos = obs['position']
#         self.agent_dir = obs['direction']
#         self.target_pos = target_pos
        
#         assert self.agent_pos[0]  < self.width and self.agent_pos[1]  < self.height
#         assert self.target_pos[0] < self.width and self.target_pos[1] < self.height
        
#         x, y = target_pos
#         self.target_pos_and_dir = [(x-1, y, 0), (x, y-1, 1), (x+1, y, 2), (x, y+1, 3)]
        
#         self.path = self.plan()
#         self.walk_step = 0  
        
#     def plan(self):
#         start_node = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
#         stop_pos = self.target_pos
        
#         open_list = set([start_node])
#         closed_list = set([])
        
#         g = {}
#         g[start_node] = 0
        
#         parents = {}
#         parents[start_node] = start_node
        
#         while len(open_list) > 0:
#             n = None
            
#             for v in open_list:
#                 if n is None or g[v] < g[n]:
#                     n = v
                    
#             if n == None:
#                 print('No path found!!!')
#                 return None
            
#             ### reconstruct and return the path when the node is the goal position
#             # if n[0] == stop_pos[0] and n[1] == stop_pos[1]:
#             if n in self.target_pos_and_dir:
#                 reconst_path = []
#                 while parents[n] != n:
#                     reconst_path.append(n)
#                     n = parents[n]
                    
#                 reconst_path.append(start_node)
#                 reconst_path.reverse()
#                 return reconst_path
                
#             for m in get_neighbors(n, self.obs):
#                 if m not in open_list and m not in closed_list:
#                     open_list.add(m)
#                     parents[m] = n
#                     g[m] = g[n] + 1
                    
#                 else:
#                     if g[m] > g[n]+1:
#                         g[m] = g[n]+1
#                         parents[m] = n
                        
#                         if m in closed_list:
#                             closed_list.remove(m)
#                             open_list.add(m)
            
#             open_list.remove(n)
#             closed_list.add(n)
            
#         print('No path found!!!')
#         return None
    
#     def step(self, obs=None):
#         if obs is not None:
#             cur_pos = obs['position']
#             cur_dir = obs['direction']
#             print(cur_pos)
#             print(obs['image'][:,:,0])
#             assert cur_pos[0] == self.path[self.walk_step][0] 
#             assert cur_pos[1] == self.path[self.walk_step][1]
#             assert cur_dir == self.path[self.walk_step][2]
        
#         next_x, next_y, next_dir = self.path[self.walk_step+1]
#         self.walk_step+=1
#         done = self.walk_step == len(self.path)-1
#         if cur_dir - next_dir == 1 or (cur_dir==0 and next_dir==3):
#             return 0, done
#         elif cur_dir - next_dir == -1 or (cur_dir==3 and next_dir==0):
#             return 1, done
#         elif cur_dir - next_dir == 0:
#             return 2, done
#         else:
#             print('No action find error!!!')
#             return None, done

        
# class Explore():
#     def __init__(self, init_obs, view_size):
#         ''' 
#             Inputs:
#                 init_obs: {'image': width x height x channel, 'position': [x,y], 'direction': , 'mission': ,}
#                 view_size: env.agent_view_size
#         '''
        
#         self.view_size = view_size
#         self.scope = self.view_size // 2
#         n_turn_left = (init_obs['direction'] + 2) // 4
#         self.plan = [0] * n_turn_left
#         self.reset_to_NW = False
    
#     def step(self, obs):
#         self.map = obs['image'][:,:,0]
#         self.agent_pos = obs['position']
#         self.agent_dir = obs['direction']
            
#         if self.agent_dir == 0:
#             fwd_obj = self.map[self.agent_pos[0] + self.scope, self.agent_pos[1]]
#         elif self.agent_dir == 2:
#             fwd_obj = self.map[self.agent_pos[0] - self.scope, self.agent_pos[1]]
#         elif self.agent_dir == 1:
#             fwd_obj = self.map[self.agent_pos[0], self.agent_pos[1] + self.scope]
#         else:
#             fwd_obj = self.map[self.agent_pos[0], self.agent_pos[1] - self.scope]
        
#         if self.reset_to_NW == False:
#             if len(self.plan) > 0: # turn left until facing west
#                 action = self.plan.pop(0)
#             elif fwd_obj == 2: # hitting a wall
#                 action = 1 # west -> north -> east
#                 if self.agent_dir == 3: # facing north
#                     self.reset_to_NW = True
#             else:
#                 action = 2
#         else:
#             if fwd_obj == 2: # hitting a wall
#                 if self.agent_dir == 0: # facing east
#                     self.plan = [2] * self.view_size + [1] # go south for self.view_size step, then turn west
#                     action = 1 # east -> south
#                 elif self.agent_dir == 2: # facing west
#                     self.plan = [2] * self.view_size + [0] # go south for self.view_size step, then turn east
#                     action = 0 # west -> south
#                 else: # facing south
#                     action = self.plan[-1]
#                     self.plan = []
#             elif len(self.plan) > 0:
#                 action = self.plan.pop(0)
#             else:
#                 action = 2
                
#         if np.count_nonzero(self.map == 0) == 0: # no unseen
#             done = True
#         else:
#             done = False
                
#         return action, done

    
# class Pickup():
#     def __init__(self):
#         pass
    
#     def step(self, obs):
#         return 3, True

    
# class Drop():
#     def __init__(self):
#         pass
    
#     def step(self, obs):
#         return 4, True

    
# class Toggle():
#     def __init__(self):
#         pass
    
#     def step(self, obs):
#         return 5, True
    
    

