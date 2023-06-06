#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Game_RL.py
@Time    :   2023/06/01 09:17:57
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

from commuication_net import RL_Net
from executive_net import Executive_net
import os, json, sys
import utils
import gymnasium as gym
import env
import algos
import torch
import numpy as np
import time
import skill
from mediator import SKILL_TO_IDX, SimpleDoorKey_Mediator
from .Game import Game
prefix = os.getcwd()
task_info_json = os.path.join(prefix, "prompt/task_info.json")
class Game_RL(Game):
    def __init__(self, args, policy=None):
        super().__init__(args, policy) 
        obs_space, preprocess_obss = utils.get_obss_preprocessor(self.env_fn().observation_space)
        self.record_frames = args.record
        if policy is None:
            ## communication network for ask
            self.RL_net = RL_Net(obs_space, 5).to(self.device)
        else:
            self.RL_net = policy.to(self.device)

        self.ppo_algo = algos.PPO(self.RL_net, device=self.device, save_path=self.logger.dir, batch_size=self.batch_size)

        self.mediator = SimpleDoorKey_Mediator(self.ideal)


    def flag2skill(self,obs, skill_flag):

        # print(text)
        goal = {}
        if skill_flag == 0:
            goal["action"] = SKILL_TO_IDX["explore"]
            goal["object"] = None
        elif skill_flag == 1:
            agent_map = obs[:, :, 3]
            agent_pos = np.argwhere(agent_map != 4)[0]
            if 'key' not in self.mediator.obj_coordinate.keys() or obs[:,:,0][agent_pos[0],agent_pos[1]] == 5:
                goal["action"] = None
            else:
                goal["action"] = SKILL_TO_IDX["go to object"]
                goal["coordinate"] = self.mediator.obj_coordinate["key"]
        elif skill_flag == 2:
            if 'door' not in self.mediator.obj_coordinate.keys():
                goal["action"] = None
            else:
                goal["action"] = SKILL_TO_IDX["go to object"]
                goal["coordinate"] = self.mediator.obj_coordinate["door"]
        elif skill_flag == 3:
            goal["action"] = SKILL_TO_IDX["pickup"]
        elif skill_flag == 4:
            goal["action"] = SKILL_TO_IDX["unlock"]
        return [goal]

    def collect(self, env_fn, seed=None):
        # Do one agent-environment interaction
        env = utils.WrapEnv(env_fn)
        with torch.no_grad():
            buffer = algos.Buffer(self.gamma, self.lam)
            obs = env.reset(seed)
            self.mediator.reset()
            done = False
        
            traj_len = 0
            pre_skill = None
            if self.frame_stack >1:
                com_obs = obs
            else:
                his_obs = obs
                com_obs = obs - his_obs

            utils.global_param.set_value('exp', None)
            utils.global_param.set_value('explore_done', False)
            while not done and traj_len < self.max_ep_len:
                dist, value = self.RL_net(torch.Tensor(com_obs).to(self.device))         
                skill_flag = dist.sample()
                log_probs = dist.log_prob(skill_flag)
                skill = self.flag2skill(obs[0],skill_flag)
                if skill != pre_skill or skill_done:
                    # print(skill)
                    self.Executive_net = Executive_net(skill,obs[0],self.agent_view_size)
                    pre_skill = skill

                ## RL choose skill, and return action_list
                action, skill_done = self.Executive_net(obs[0])
                # print("last action", action)
                ## one step do one action in action_list
                next_obs, reward, done, info = env.step(np.array([action]))
                # print("next_obs\n", next_obs[0,:,:,4].T)
                # print("next_obs_pos\n", next_obs[0,:,:,7].T)    
                # print("done?", done) 
    
                buffer.store(com_obs, skill_flag.to("cpu").numpy(), reward, value.to("cpu").numpy(), log_probs.to("cpu").numpy()) 
                if self.frame_stack >1:
                    obs = next_obs
                    com_obs = obs
                else:
                    his_obs = obs
                    obs = next_obs
                    com_obs = obs - his_obs
                traj_len += 1
            _, value = self.RL_net(torch.Tensor(com_obs).to(self.device))
            buffer.finish_path(last_val=(not done) * value.to("cpu").numpy())
        return buffer




    def eval(self, env_fn, trajs=1, seed=None, show_dialogue=False):
        env = utils.WrapEnv(env_fn)
        with torch.no_grad():
            ep_returns = []
            ep_lens = []
            for traj in range(trajs):
                obs = env.reset(seed)
                self.mediator.reset()
                if self.record_frames:
                    video_frames = [obs['rgb']]
                    goal_frames = ["start"] 

                done = False
                skill_done = True
                traj_len = 0
                ep_return = 0
                pre_skill = None
                if self.frame_stack > 1:
                    com_obs = obs
                else:
                    his_obs = obs
                    com_obs = obs - his_obs
                utils.global_param.set_value('exp', None)
                utils.global_param.set_value('explore_done', False)
                while not done and traj_len < self.max_ep_len:
                    dist, _ = self.RL_net(torch.Tensor(com_obs).to(self.device))
                    skill_flag = dist.sample()

                    skill = self.flag2skill(obs[0],skill_flag)
                    if skill != pre_skill or skill_done:
                        self.Executive_net = Executive_net(skill,obs[0],self.agent_view_size)
                        pre_skill = skill
                    
                    ## RL choose skill, and return action_list
                    action, skill_done = self.Executive_net(obs[0])
                    ## one step do one action in action_list
                    next_obs, reward, done, info = env.step(np.array([action]))
                    if self.frame_stack >1:
                        obs = next_obs
                        com_obs = obs
                    else:
                        his_obs = obs
                        obs = next_obs
                        com_obs = obs - his_obs
            
                    ep_return += 1.0*reward
                    traj_len += 1
                if show_dialogue:
                    if done:
                        print("RL: Task Completed! \n")
                    else:
                        print("RL: Task Fail! \n")
                ep_returns += [ep_return]
                ep_lens += [traj_len]
   
            return np.mean(ep_returns), np.mean(ep_lens), 0., np.mean(ep_returns)

 
if __name__ == '__main__':
   pass

