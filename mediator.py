#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mediator.py
@Time    :   2023/05/16 10:22:36
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import numpy as np
import re
import copy
from abc import ABC, abstractmethod
@staticmethod
def get_minigrid_words():
    colors = ["red", "green", "blue", "yellow", "purple", "grey"]
    objects = [
        "unseen",
        "empty",
        "wall",
        "floor",
        "box",
        "key",
        "ball",
        "door",
        "goal",
        "agent",
        "lava",
    ]

    verbs = [
        "pick",
        "avoid",
        "get",
        "find",
        "put",
        "use",
        "open",
        "go",
        "fetch",
        "reach",
        "unlock",
        "traverse",
    ]

    extra_words = [
        "up",
        "the",
        "a",
        "at",
        ",",
        "square",
        "and",
        "then",
        "to",
        "of",
        "rooms",
        "near",
        "opening",
        "must",
        "you",
        "matching",
        "end",
        "hallway",
        "object",
        "from",
        "room",
    ]

    all_words = colors + objects + verbs + extra_words
    assert len(all_words) == len(set(all_words))
    return {word: i for i, word in enumerate(all_words)}

# Map of agent direction, 0: East; 1: South; 2: West; 3: North
DIRECTION = {
    0: [1, 0],
    1: [0, 1],
    2: [-1, 0],
    3: [0, -1],
}

# Map of object type to integers
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
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

# Map of skill names to integers
SKILL_TO_IDX = {"explore": 0, "go to object": 1, "pickup": 2, "drop": 3, "toggle": 4}
IDX_TO_SKILL = dict(zip(SKILL_TO_IDX.values(), SKILL_TO_IDX.keys()))




class Base_Mediator(ABC):
    """The base class for Base_Mediator."""

    def __init__(self):
        super().__init__()
        self.obj_coordinate = {}

    @abstractmethod
    def RL2LLM(self):
        pass
    @abstractmethod
    def LLM2RL(self):
        pass
    
    def reset(self):
        self.obj_coordinate = {}

class SimpleDoorKey_Mediator(Base_Mediator):
    def __init__(self, ):
        super().__init__()

    # ## obs2text
    def RL2LLM(self, obs):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)
        carry_object = None
        if len(key_list):
            for key in key_list:
                i, j = key
                color = obs[i,j,1]
                #object = f"{IDX_TO_COLOR[color]} key"
                object = "key"
                if (agent_pos == key).all():
                    carry_object = object
                else:
                    if context != "":
                        context += ", "
                    context += f"observed a {object}" 
                    self.obj_coordinate[object] = (i,j)

        if len(door_list):
            for door in door_list:
                i, j = door
                color = obs[i,j,1]
                #object = f"{IDX_TO_COLOR[color]} door"
                object = "door"
                if context != "":
                    context += ", "
                context += f"observed a {object}"
                self.obj_coordinate[object] = (i,j)

        if context == '':
            context += "observed {}".format('nothing')
        if carry_object is not None:
            if context != "":
                context += ", "
            context += f"carry {carry_object}"
        context = f"observation: {{{context}}} "
        return context
    
    def parser(self, text):
        # action:
        if "explore" in text:
            act = SKILL_TO_IDX["explore"]
        elif "go to" in text:
            act = SKILL_TO_IDX["go to object"]
        elif "pick up" in text:
            act = SKILL_TO_IDX["pickup"]
        elif "drop" in text:
            act = SKILL_TO_IDX["drop"]
        elif "toggle" in text:
            act = SKILL_TO_IDX["toggle"]
        else:
            print("Unknown Planning :", text)
            act = 6 # do nothing
        # object:
        try:
            if "key" in text:
                obj = OBJECT_TO_IDX["key"]
                coordinate = self.obj_coordinate["key"]
            elif "door" in text:
                obj = OBJECT_TO_IDX["door"]
                coordinate = self.obj_coordinate["door"]
            else:
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None
        except:
            print("Unknown Planning :", text)
            act = 6 # do nothing
        return act, obj, coordinate
    
    
    def LLM2RL(self, plan):

        plan = re.findall(r'{(.*?)}', plan)
        lines = plan[0].split(',')
        skill_list = []
        for line in lines:

            action, object, coordinate = self.parser(line)
            goal = {}
            goal["action"] = action
            goal["object"] = object
            goal["coordinate"] = coordinate
            skill_list.append(goal)
        return skill_list
    
class KeyInBox_Mediator(Base_Mediator):
    def __init__(self):
        super().__init__()
       
    # ## obs2text
    def RL2LLM(self, obs):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)
        box_list = np.argwhere(obs_object==7)
        carry_object = None
        if len(key_list):
            for key in key_list:
                i, j = key
                object = "key"
                if (agent_pos == key).all():
                    carry_object = object
                else:
                    if context != "":
                        context += ", "
                    context += f"observed a {object}" 
                    self.obj_coordinate[object] = (i,j)
        if len(box_list):
            for box in box_list:
                i, j = box
                object = "box"
                if context != "":
                    context += ", "
                context += f"observed a {object}" 
                self.obj_coordinate[object] = (i,j)
        if len(door_list):
            for door in door_list:
                i, j = door
                object = "door"
                if context != "":
                    context += ", "
                context += f"observed a {object}"
                self.obj_coordinate[object] = (i,j)
                
        if context == '':
            context += "observed {}".format('nothing')
        context = f"observation: {{{context}}}, "
        if carry_object is not None:
            if context != "":
                context += ", "
            context += f"carry {carry_object}"
        return context
    
    def parser(self, text):
        # action:
        if "explore" in text:
            act = SKILL_TO_IDX["explore"]
        elif "go to" in text:
            act = SKILL_TO_IDX["go to object"]
        elif "pick up" in text:
            act = SKILL_TO_IDX["pickup"]
        elif "drop" in text:
            act = SKILL_TO_IDX["drop"]
        elif "toggle" in text:
            act = SKILL_TO_IDX["toggle"]
        else:
            print("Unknown Planning :", text)
            act = 6 # do nothing
        # object:
        try:
            if "key" in text:
                obj = OBJECT_TO_IDX["key"]
                if "key" in self.obj_coordinate.keys():
                    coordinate = self.obj_coordinate["key"]
                else:
                    coordinate = self.obj_coordinate["box"]
            elif "door" in text:
                obj = OBJECT_TO_IDX["door"]
                coordinate = self.obj_coordinate["door"]
            elif "box" in text:
                obj = OBJECT_TO_IDX["box"]
                coordinate = self.obj_coordinate["box"]
            else:
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None
        except:
            print("Unknown Planning :", text)
            act = 6 # do nothing
        return act, obj, coordinate
    
    
    def LLM2RL(self, plan):
        plan = re.findall(r'{(.*?)}', plan)
        lines = plan[0].split(',')
        skill_list = []
        for line in lines:

            action, object, coordinate = self.parser(line)
            goal = {}
            goal["action"] = action
            goal["object"] = object
            goal["coordinate"] = coordinate
            skill_list.append(goal)
        return skill_list
    
class RandomBoxKey_Mediator(Base_Mediator):
    def __init__(self):
        super().__init__()
      
    # ## obs2text
    def RL2LLM(self, obs):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)
        box_list = np.argwhere(obs_object==7)
        carry_object = None
        if len(key_list):
            for key in key_list:
                i, j = key
                object = "key"
                if (agent_pos == key).all():
                    carry_object = object
                else:
                    if context != "":
                        context += ", "
                    context += f"observed a {object}" 
                    self.obj_coordinate[object] = (i,j)
        if len(box_list):
            for box in box_list:
                i, j = box
                object = "box"
                if context != "":
                    context += ", "
                context += f"observed a {object}" 
                self.obj_coordinate[object] = (i,j)
        if len(door_list):
            for door in door_list:
                i, j = door
                object = "door"
                if context != "":
                    context += ", "
                context += f"observed a {object}"
                self.obj_coordinate[object] = (i,j)

        if context == '':
            context += "observed {}".format('nothing')
        if carry_object is not None:
            if context != "":
                context += ", "
            context += f"carry {carry_object}"
        context = f"observation: {{{context}}} "
        return context
    
    def parser(self, text):
        # action:
        if "explore" in text:
            act = SKILL_TO_IDX["explore"]
        elif "go to" in text:
            act = SKILL_TO_IDX["go to object"]
        elif "pick up" in text:
            act = SKILL_TO_IDX["pickup"]
        elif "drop" in text:
            act = SKILL_TO_IDX["drop"]
        elif "toggle" in text:
            act = SKILL_TO_IDX["toggle"]
        else:
            print("Unknown Planning :", text)
            act = 6 # do nothing
        # object:
        try:
            if "key" in text:
                obj = OBJECT_TO_IDX["key"]
                if "key" in self.obj_coordinate.keys():
                    coordinate = self.obj_coordinate["key"]
                else:
                    coordinate = self.obj_coordinate["box"]
            elif "door" in text:
                obj = OBJECT_TO_IDX["door"]
                coordinate = self.obj_coordinate["door"]
            elif "box" in text:
                obj = OBJECT_TO_IDX["box"]
                coordinate = self.obj_coordinate["box"]
            else:
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None
        except:
            print("Unknown Planning :", text)
            act = 6 # do nothing
        return act, obj, coordinate
    
    
    def LLM2RL(self, plan):
        plan = re.findall(r'{(.*?)}', plan)
        lines = plan[0].split(',')
        skill_list = []
        for line in lines:

            action, object, coordinate = self.parser(line)
            goal = {}
            goal["action"] = action
            goal["object"] = object
            goal["coordinate"] = coordinate
            skill_list.append(goal)
        return skill_list

class ColoredDoorKey_Mediator(Base_Mediator):
    def __init__(self):
        super().__init__()
     
    # ## obs2text
    def RL2LLM(self, obs):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]
        agent_dir = agent_map[agent_pos[0],agent_pos[1]]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)

        block_object = None
        carry_object = None
        if len(key_list):
            for key in key_list:
                i, j = key
                color = obs[i,j,1]
                object = f"{IDX_TO_COLOR[color]} key"

                if (agent_pos == key).all():
                    carry_object = object
                else:
                    if context != "":
                        context += ", "
                    context += f"observed {object}" 
                    self.obj_coordinate[object] = (i,j)

                    if (agent_pos + DIRECTION[agent_dir] == key).all():
                        block_object = object
        if len(door_list):
            for door in door_list:
                i, j = door
                color = obs[i,j,1]
                object = f"{IDX_TO_COLOR[color]} door"
                if context != "":
                    context += ", "
                context += f"observed {object}"
                self.obj_coordinate[object] = (i,j)

        # if block_object is not None:
        #     if context != "":
        #         context += ", "
        #     context += f"block by {block_object}"

        if context == '':
            context += "observed {}".format('nothing')
        if carry_object is not None:
            if context != "":
                context += ", "
            context += f"carry {carry_object}"
        #context = f"observation:{{{context}}},"
        context = f"Q: [{context}]"
        return context
    
    def parser(self, text):
        # action:
        if "explore" in text:
            act = SKILL_TO_IDX["explore"]
        elif "go to" in text:
            act = SKILL_TO_IDX["go to object"]
        elif "pick up" in text:
            act = SKILL_TO_IDX["pickup"]
        elif "drop" in text:
            act = SKILL_TO_IDX["drop"]
        elif "toggle" in text:
            act = SKILL_TO_IDX["toggle"]
        else:
            print("Unknown Planning :", text)
            act = 6 # do nothing
        # object:
        try:
            if "key" in text:
                obj = OBJECT_TO_IDX["key"]    
                words = text.split(' ')
                filter_words = []
                for w in words:
                    w1="".join(c for c in w if c.isalpha())
                    filter_words.append(w1)
                object_word = filter_words[-2] + " " + filter_words[-1]
                coordinate = self.obj_coordinate[object_word]
            elif "door" in text:
                obj = OBJECT_TO_IDX["door"]
                words = text.split(' ')
                filter_words = []
                for w in words:
                    w1="".join(c for c in w if c.isalpha())
                    filter_words.append(w1)
                object_word = filter_words[-2] + " " + filter_words[-1]
                coordinate = self.obj_coordinate[object_word]
            else:
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None
        except:
            print("Unknown Planning :", text)
            coordinate = None
            act = 6 # do nothing
        return act, obj, coordinate
    
    
    def LLM2RL(self, plan):
        plan = re.findall(r'{(.*?)}', plan)
        lines = plan[0].split(',')
        skill_list = []
        for line in lines:

            action, object, coordinate = self.parser(line)
            goal = {}
            goal["action"] = action
            goal["object"] = object
            goal["coordinate"] = coordinate
            skill_list.append(goal)
        return skill_list


if __name__ == "__main__":
    word = get_minigrid_words()