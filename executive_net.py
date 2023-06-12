import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from skill import GoTo_Goal, Explore, Pickup, Toggle
from utils import global_param
from mediator import IDX_TO_SKILL, IDX_TO_OBJECT

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
        
    @property
    def current_skill(self):
        skill = self.skill_list[self.current_index]
        return IDX_TO_SKILL[skill['action']] + ' ' + IDX_TO_OBJECT[skill['object']]

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


