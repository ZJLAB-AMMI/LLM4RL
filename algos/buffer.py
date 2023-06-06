#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   buffer.py
@Time    :   2023/04/19 09:40:36
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch


def Merge_Buffers(buffers, device='cpu'):
    merged = Buffer(device=device)
    for buf in buffers:
        offset = len(merged)

        merged.states  += buf.states
        merged.actions += buf.actions
        merged.rewards += buf.rewards
        merged.values  += buf.values
        merged.returns += buf.returns
        merged.log_probs += buf.log_probs
        

        merged.ep_returns += buf.ep_returns
        merged.ep_lens    += buf.ep_lens
        merged.ep_game_returns += buf.ep_game_returns
        merged.ep_interactions += buf.ep_interactions


        merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
        merged.ptr += buf.ptr

    return merged


class Buffer:
    """
    A buffer for storing trajectory data and calculating returns for the policy
    and critic updates.
    """
    def __init__(self, gamma=0.99, lam=0.95, device='cpu'):
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.returns = []
        self.log_probs = []
        self.game_rewards = [] # without communication penalty


        self.ep_returns = [] # for logging
        self.ep_lens    = []
        self.ep_game_returns = [] # without communication penalty
        self.ep_interactions = [] # interactions number


        self.gamma, self.lam = gamma, lam
        self.device = device

        self.ptr = 0
        self.traj_idx = [0]

    def __len__(self):
        return self.ptr

    def store(self, state, action, reward, value, log_probs, game_reward=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        self.states  += [state.squeeze(0)]
        self.actions += [action.squeeze(0)]
        self.rewards += [reward.squeeze(0)]
        self.values  += [value.squeeze(0)]
        self.log_probs += [log_probs.squeeze(0)]
        if game_reward is not None:
            self.game_rewards += [game_reward.squeeze(0)]
        self.ptr += 1

    def finish_path(self, last_val=None, interactions=None):
        self.traj_idx += [self.ptr]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]
        
  

        returns = []

        R = last_val.squeeze(0).copy()  # Avoid copy?
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R) 

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        if self.game_rewards != []:
            game_rewards = self.game_rewards[self.traj_idx[-2]:self.traj_idx[-1]]
            self.ep_game_returns += [np.sum(game_rewards)]

        self.ep_lens    += [len(rewards)]
        self.ep_interactions += [interactions]
        

    
    def get(self):
        return(
            np.array(self.states),
            np.array(self.actions),
            np.array(self.returns),
            np.array(self.values),
            np.array(self.log_probs)
        )

    def sample(self, batch_size=64, recurrent=False):
        if recurrent:
            random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
            sampler = BatchSampler(random_indices, batch_size, drop_last=False)
        else:
            random_indices = SubsetRandomSampler(range(self.ptr))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)

        observations, actions, returns, values, log_probs = map(torch.Tensor, self.get())

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for indices in sampler:
            if recurrent:
                obs_batch       = [observations[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                action_batch    = [actions[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                return_batch    = [returns[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                advantage_batch = [advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                values_batch    = [values[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                mask            = [torch.ones_like(r) for r in return_batch]
                log_prob_batch  = [log_probs[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]

                obs_batch       = pad_sequence(obs_batch, batch_first=False)
                action_batch    = pad_sequence(action_batch, batch_first=False)
                return_batch    = pad_sequence(return_batch, batch_first=False)
                advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                values_batch    = pad_sequence(values_batch, batch_first=False)
                mask            = pad_sequence(mask, batch_first=False)
                log_prob_batch  = pad_sequence(log_prob_batch, batch_first=False)
            else:
                obs_batch       = observations[indices]
                action_batch    = actions[indices]
                return_batch    = returns[indices]
                advantage_batch = advantages[indices]
                values_batch    = values[indices]
                mask            = torch.FloatTensor([1])
                log_prob_batch  = log_probs[indices]


            yield obs_batch.to(self.device), action_batch.to(self.device), return_batch.to(self.device), advantage_batch.to(self.device), values_batch.to(self.device), mask.to(self.device), log_prob_batch.to(self.device)


