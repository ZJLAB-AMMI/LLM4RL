#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   collect.py
@Time    :   2023/04/19 10:14:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

#import ray
from copy import deepcopy
import torch
import time
from algos.buffer import Buffer
import numpy as np
from utils.env import WrapEnv

# @ray.remote(num_gpus=1)
class Collect_Worker:
    def __init__(self, policy, critic, device, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.policy = deepcopy(policy)
        self.critic = deepcopy(critic)
        self.device = device


    def sync_policy(self, new_actor_params, new_critic_params):
        for p, new_p in zip(self.policy.parameters(), new_actor_params):
            p.data.copy_(new_p)

        for p, new_p in zip(self.critic.parameters(), new_critic_params):
            p.data.copy_(new_p)


    def collect(self, max_traj_len, min_steps, env_fn, anneal=1.0):
        env = WrapEnv(env_fn)
        with torch.no_grad():
            memory = Buffer(self.gamma, self.lam)
            num_steps = 0
            # state = env.reset()
            while num_steps < min_steps:
                state = torch.Tensor(env.reset())
                done = False
                value = 0
                traj_len = 0

                if hasattr(self.policy, 'init_hidden_state'):
                    self.policy.init_hidden_state()

                if hasattr(self.critic, 'init_hidden_state'):
                    self.critic.init_hidden_state()

                while not done and traj_len < max_traj_len:
                    action = self.policy(state.to(self.device), deterministic=False, anneal=anneal).to("cpu")
                    value = self.critic(state.to(self.device)).to("cpu")

                    next_state, reward, done, _ = env.step(action.numpy())
                    memory.store(state.numpy(), action.numpy(), reward, value.numpy())

                    state = torch.Tensor(next_state)
                    traj_len += 1
                    num_steps += 1

                value = self.critic(state.to(self.device)).to("cpu")
                memory.finish_path(last_val=(not done) * value.numpy())

            return memory
        
    def evaluate(self, max_traj_len, env_fn, trajs=1):
        torch.set_num_threads(1)
        env = WrapEnv(env_fn)
        with torch.no_grad():
            ep_returns = []
            for traj in range(trajs):
                state = torch.Tensor(env.reset())
                done = False
                traj_len = 0
                ep_return = 0

                if hasattr(self.policy, 'init_hidden_state'):
                    self.policy.init_hidden_state()

                while not done and traj_len < max_traj_len:
                    action = self.policy(state.to(self.device), deterministic=False, anneal=1.0).to("cpu")

                    next_state, reward, done, _ = env.step(action.numpy())

                    state = torch.Tensor(next_state)
                    ep_return += reward
                    traj_len += 1
                ep_returns += [ep_return]
            return np.mean(ep_returns)

