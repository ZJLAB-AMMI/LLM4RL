#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   planner.py
@Time    :   2023/05/16 09:12:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


import os, requests
from typing import Any
from mediator import *
from utils import global_param


from abc import ABC, abstractmethod

class Base_Planner(ABC):
    """The base class for Planner."""

    def __init__(self):
        super().__init__()
        self.dialogue_system = ''                  
        self.dialogue_user = ''
        self.dialogue_logger = ''         
        self.show_dialogue = False
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show

    ## initial prompt, write in 'prompt/task_info.json
    def initial_planning(self, decription, example):

        prompts = decription + example
        self.dialogue_system += decription + "\n"
        self.dialogue_system += example + "\n"

        ## set system part
        server_error_cnt = 0
        while server_error_cnt<10:
            try:
                url = 'http://10.106.27.11:8000/v1/chat/completions'
                headers = {'Content-Type': 'application/json'}
                
                data = {'model': "vicuna-7b-1", "messages":[{"role": "system", "content": prompts}]}
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()                    
                    server_flag = 1
                                
                   
                if server_flag:
                    break
                    
            except Exception as e:
                server_error_cnt += 1
                print(e)    

    def query_codex(self, prompt_text):
        server_flag = 0
        server_error_cnt = 0
        response = ''
        while server_error_cnt<10:
            try:
                #response =  openai.Completion.create(prompt_text)
                url = 'http://10.106.27.11:8000/v1/chat/completions'
                headers = {'Content-Type': 'application/json'}
                
                # prompt_text
                
                data = {'model': "vicuna-7b-1", "messages":[{"role": "user", "content": prompt_text }]}
                response = requests.post(url, headers=headers, json=data)
                
                

                if response.status_code == 200:
                    result = response.json()                    
                    server_flag = 1
                                
                   
                if server_flag:
                    break
                    
            except Exception as e:
                server_error_cnt += 1
                print(e)
        if result is None:
            return
        else:
            return result['messages'][-1][-1] 

    def check_plan_isValid(self, plan):
        if "{" in plan and "}" in plan:
            return True
        else:
            return False
        
    def step_planning(self, text):
        ## seed for LLM and get feedback
        plan = self.query_codex(text)
        if plan is not None:
            ## check Valid, llm may give wrong answer
            while not self.check_plan_isValid(plan):
                print("%s is illegal Plan! Replan ...\n" %plan)
                plan = self.query_codex(text)
        return plan

    @abstractmethod
    def forward(self):
        pass

class SimpleDoorKey_Planner(Base_Planner):
    def __init__(self):
        super().__init__()
        
        self.mediator = SimpleDoorKey_Mediator()

    def __call__(self, input):
        return self.forward(input)
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)
      
        self.step_planning("reset")

    def forward(self, obs):
        text = self.mediator.RL2LLM(obs)
        # print(text)
        plan = self.step_planning(text)
        
        self.dialogue_logger += text
        self.dialogue_logger += plan
        self.dialogue_user = text +"\n"
        self.dialogue_user += plan
        if self.show_dialogue:
            print(self.dialogue_user)
        skill = self.mediator.LLM2RL(plan)
        return skill
   
    
class KeyInBox_Planner(Base_Planner):
    def __init__(self):
        super().__init__()
        self.mediator = KeyInBox_Mediator()

    def __call__(self, input):
        return self.forward(input)
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)

        self.step_planning("reset")

    def forward(self, obs):
        text = self.mediator.RL2LLM(obs)
        # print(text)
        plan = self.step_planning(text)

        self.dialogue_logger += text
        self.dialogue_logger += plan
        self.dialogue_user = text +"\n"
        self.dialogue_user += plan
        if self.show_dialogue:
            print(self.dialogue_user)
        skill = self.mediator.LLM2RL(plan)
        return skill

from torch.distributions.categorical import Categorical
import torch
class RandomBoxKey_Planner(Base_Planner):
    def __init__(self):
        super().__init__()
        self.mediator = RandomBoxKey_Mediator()
     
    def __call__(self, input):
        return self.forward(input)
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)
        self.step_planning("reset")
     
    def forward(self, obs):
        text = self.mediator.RL2LLM(obs)
        # print(text)
        plan = self.step_planning(text)
        
        self.dialogue_logger += text
        self.dialogue_logger += plan
        self.dialogue_user = text +"\n"
        self.dialogue_user += plan
        if self.show_dialogue:
            print(self.dialogue_user)
        skill = self.mediator.LLM2RL(plan)
        return skill
   
class ColoredDoorKey_Planner(Base_Planner):
    def __init__(self):
        super().__init__()
        self.mediator = ColoredDoorKey_Mediator()

    def __call__(self, input):
        return self.forward(input)
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)
    
        self.step_planning("reset")

    def forward(self, obs):
        text = self.mediator.RL2LLM(obs)
        # print(text)
        plan = self.step_planning(text)

        self.dialogue_logger += text
        self.dialogue_logger += plan
        self.dialogue_user = text +"\n"
        self.dialogue_user += plan
        if self.show_dialogue:
            print(self.dialogue_user)
        skill = self.mediator.LLM2RL(plan)
        return skill
   

def Planner(task):
    if task.lower() == "simpledoorkey":
        planner = SimpleDoorKey_Planner()
    elif task.lower() == "keyinbox":
        planner = KeyInBox_Planner()
    elif task.lower() == "randomboxkey":
        planner = RandomBoxKey_Planner()
    elif task.lower() == "coloreddoorkey":
        planner = ColoredDoorKey_Planner()
    return planner
