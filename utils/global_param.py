#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   global_param.py
@Time    :   2023/05/23 10:55:41
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   global parameter dictory for skill 
'''

def init():
    global _global_dict
    _global_dict = {}

def set_value(key, value):
    _global_dict[key] = value

def get_value(key, defValue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defValue