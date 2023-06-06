#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base.py
@Time    :   2023/05/18 09:42:14
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


from abc import ABC, abstractmethod

class Base(ABC):
    """The base class for RL algorithms."""

    def __init__(self, model, device, num_steps, lr, max_grad_norm, entropy_coef, value_loss_coef, num_worker):
        super().__init__()
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_worker = num_worker

        # self.model.to(self.device)
        # self.model.train()

    @abstractmethod
    def update_policy(self):
        pass