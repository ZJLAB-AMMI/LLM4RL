#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ppo.py
@Time    :   2023/05/17 11:23:57
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

from .base import Base
import numpy as np
import torch
import torch.nn.functional as F
import os

class PPO(Base):

    def __init__(self, model, device=None, save_path=None, num_steps=4096, lr=0.001, max_grad_norm=0.5, num_worker=4, epoch=3, batch_size=64, adam_eps=1e-8,entropy_coef=0.01,value_loss_coef=0.5,clip_eps=0.2,recurrent=False):
        super().__init__(model, device, num_steps, lr, max_grad_norm, entropy_coef, value_loss_coef, num_worker)

        self.lr             = lr
        self.eps            = adam_eps
        self.entropy_coeff  = entropy_coef
        self.value_loss_coef= value_loss_coef
        self.clip           = clip_eps
        self.minibatch_size = batch_size
        self.epochs         = epoch
        self.num_steps      = num_steps
        self.num_worker     = num_worker
        self.grad_clip      = max_grad_norm
        self.recurrent      = recurrent
        self.device         = device

        self.model = model.to(self.device)
        self.save_path = save_path

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, eps=self.eps)

    def save(self):
        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch model
        torch.save(self.model, os.path.join(self.save_path, "acmodel" + filetype))
        #torch.save(self.policy, os.path.join(self.save_path, "actor" + filetype))
        #torch.save(self.critic, os.path.join(self.save_path, "critic" + filetype))

    def update_policy(self, buffer):
        losses = []
        for _ in range(self.epochs):
            for batch in buffer.sample(self.minibatch_size, self.recurrent):
                obs_batch, action_batch, return_batch, advantage_batch, values_batch,  mask, log_prob_batch = batch
                pdf, value = self.model(obs_batch)

                entropy_loss = (pdf.entropy() * mask).mean()

                ratio = torch.exp(pdf.log_prob(action_batch) - log_prob_batch)
                surr1 = ratio * advantage_batch * mask
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = values_batch + torch.clamp(value - values_batch, -self.clip, self.clip)
                surr1 = ((value - return_batch)*mask).pow(2)
                surr2 = ((value_clipped - return_batch)*mask).pow(2)
                value_loss = torch.max(surr1, surr2).mean()

                loss = policy_loss - self.entropy_coef * entropy_loss + self.value_loss_coef * value_loss

                # Update actor-critic
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                # Update log values
                losses.append([loss.item(), pdf.entropy().mean().item()])
        mean_losses = np.mean(losses, axis=0)
        return mean_losses
