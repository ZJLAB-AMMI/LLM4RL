#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   commuication_net.py
@Time    :   2023/05/16 16:34:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np
class Communication_Net(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        
        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(obs_space['image'][-1], 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space)
        )
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, deterministic=True):
        x = obs.transpose(1, 3).transpose(2, 3)
        
        ## only use image channel 
        #x = x[:,[0],:,:]
        x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)
        #x = self.image_fl(x)
        embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

class Multi_head_Communication_Net(Communication_Net):
    def __init__(self, obs_space, action_space, heads):
        super().__init__(obs_space, action_space)

        self.heads = heads
        self.hidden = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
        )
        # Define multi-heads actor's model
        self.multi_heads_actor = []
        for l in range(self.heads):          
            actor_head = nn.Linear(64, action_space)
            self.multi_heads_actor.append(actor_head)

    def forward(self, obs_skill, deterministic=True):
        skill = []
        obs = []
        for i, os in enumerate(obs_skill):
            skill.append(os[-1])
            obs.append(os[0])
        obs = torch.stack(obs)
        skill = np.array(skill)
        x = obs.transpose(1, 3).transpose(2, 3)
        
        ## only use image channel 
        #x = x[:,[0],:,:]
        x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        embedding = x
        hidden = self.hidden(embedding)

        output = []
        for i, actor in enumerate(self.multi_heads_actor):
            x = actor.to(obs.device)(hidden)
            #dist = Categorical(logits=F.log_softmax(x, dim=1))
            output.append(x)
        
        x = self.critic(embedding)
        value = x.squeeze(1)

        dist_x = torch.zeros(len(skill), 2).to(embedding.device)
        for i in range(len(skill)):
            if skill[i] is None:
                dist_choose = np.random.choice(self.heads)
            elif skill[i][0]['action'] == 0:
                dist_choose = 0
            elif skill[i][0]['action'] == 1:
                dist_choose = 1
            elif skill[i][0]['action'] == 2:
                dist_choose = 2
            elif skill[i][0]['action'] == 4:
                dist_choose = 3
            dist_x[i] = output[dist_choose][0]
        dist = Categorical(logits=F.log_softmax(dist_x, dim=1))
        return dist, value

         


class RL_Net(nn.Module):
    """ select skill by RL instand of LLM 
        Action 0: Explore
        Action 1: go to key
        Action 2: go to door
        Action 3: pickup key
        Action 4: toggle door
    """
    def __init__(self, obs_space, action_space):
        super().__init__()

        
        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(obs_space['image'][-1], 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.image_fl = nn.Linear(n*m, self.embedding_size)
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space)
        )
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, deterministic=True):
        x = obs.transpose(1, 3).transpose(2, 3)
        
        ## only use image channel 
        #x = x[:,[0],:,:]
        x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)
        #x = self.image_fl(x)
        embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value
    


if __name__ == '__main__':
    pass
