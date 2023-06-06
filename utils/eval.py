#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   eval.py
@Time    :   2023/05/24 09:51:17
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import env
import numpy as np
class Eval():
   def __init__(self, args, policy=None):
      self.args = args
      self.policy = policy


   def eval_policy(self, test_num):

      print("env name: %s for %s" %(self.args.task, self.args.save_name))
      game = env.Game(self.args,self.policy)
      game.reset()

      reward = []
      game_reward = []
      lens = []
      interactions = []
      fail = 0
      for i in range(test_num):
         ## eval_policy ##
         if game.seed == None:
            eval_reward, eval_len, eval_interactions, eval_game_reward = game.eval(game.env_fn, seed = i, show_dialogue=game.show_dialogue)
         else:
            eval_reward, eval_len, eval_interactions, eval_game_reward = game.eval(game.env_fn, seed = game.seed, show_dialogue=game.show_dialogue)
         print("task %s, reward %s, len %s, interaction %s, reward w/o comm penalty %s" %(i, eval_reward,eval_len, eval_interactions, eval_game_reward))
         # if eval_reward <= 0:
         if eval_len == 100: 
               fail += 1

         reward.append(eval_reward)
         game_reward.append(eval_game_reward)
         lens.append(eval_len)
         interactions.append(eval_interactions)

      print("Mean reward:", np.mean(reward))
      print("Mean reward w/o comm penalty:", np.mean(game_reward))
      print("Mean len:", np.mean(lens))
      print("Mean interactions:", np.mean(interactions))
      print("Planning success rate:", 1.- fail/test_num)

   def eval_RL_policy(self, test_num):

      print("env name: %s for %s" %(self.args.task, self.args.save_name))
      game = env.Game_RL(self.args,self.policy)
      game.reset()

      reward = []
      lens = []

      fail = 0
      for i in range(test_num):
         ## eval_policy ##
         if game.seed == None:
            eval_reward, eval_len,_, _ = game.eval(game.env_fn, seed = i, show_dialogue=game.show_dialogue)
         else:
            eval_reward, eval_len,_,_ = game.eval(game.env_fn, seed = game.seed, show_dialogue=game.show_dialogue)
         print("task %s, reward %s, len %s" %(i, eval_reward,eval_len))
         # if eval_reward <= 0:
         if eval_len == 100: 
               fail += 1

         reward.append(eval_reward)
         lens.append(eval_len)

      print("Mean reward:", np.mean(reward))
      print("Mean len:", np.mean(lens))
      print("Planning success rate:", 1.- fail/test_num)

   def eval_multi_heads_policy(self, test_num):

      print("env name: %s for %s" %(self.args.task, self.args.save_name))
      game = env.Game_multi_heads(self.args,self.policy)
      game.reset()

      reward = []
      game_reward = []
      lens = []
      interactions = []
      fail = 0
      for i in range(test_num):
         ## eval_policy ##
         if game.seed == None:
            eval_reward, eval_len, eval_interactions, eval_game_reward = game.eval(game.env_fn, seed = i, show_dialogue=game.show_dialogue)
         else:
            eval_reward, eval_len, eval_interactions, eval_game_reward = game.eval(game.env_fn, seed = game.seed, show_dialogue=game.show_dialogue)
         print("task %s, reward %s, len %s, interaction %s, reward w/o comm penalty %s" %(i, eval_reward,eval_len, eval_interactions, eval_game_reward))
         # if eval_reward <= 0:
         if eval_len == 100: 
               fail += 1

         reward.append(eval_reward)
         game_reward.append(eval_game_reward)
         lens.append(eval_len)
         interactions.append(eval_interactions)
 

      print("Mean reward:", np.mean(reward))
      print("Mean reward w/o comm penalty:", np.mean(game_reward))
      print("Mean len:", np.mean(lens))
      print("Mean interactions:", np.mean(interactions))
      print("Planning success rate:", 1.- fail/test_num)

   def eval_baseline(self, test_num):
      print("env name: %s for %s" %(self.args.task, "baseline"))
      game = env.Game(self.args)
      game.reset()
      reward = []
      game_reward = []
      lens = []
      interactions = []
      fail = 0
      for i in range(test_num):
         ## eval_policy ##
         if game.seed == None:
            eval_reward, eval_len, eval_interactions, eval_game_reward = game.baseline_eval(game.env_fn, seed = i, show_dialogue=game.show_dialogue)
         else:
            eval_reward, eval_len, eval_interactions, eval_game_reward = game.baseline_eval(game.env_fn, seed = game.seed, show_dialogue=game.show_dialogue)
         print("task %s, reward %s, len %s, interaction %s, reward w/o comm penalty %s" %(i, eval_reward,eval_len, eval_interactions, eval_game_reward))
         # if eval_reward <= 0:
         if eval_len == 100: 
               fail += 1

         reward.append(eval_reward)
         game_reward.append(eval_game_reward)
         lens.append(eval_len)
         interactions.append(eval_interactions)

      print("Mean reward:", np.mean(reward))
      print("Mean reward w/o comm penalty:", np.mean(game_reward))
      print("Mean len:", np.mean(lens))
      print("Mean interactions:", np.mean(interactions))
      print("Planning success rate:", 1.- fail/test_num)

   def eval_always_ask(self, test_num):
      print("env name: %s for %s" %(self.args.task, "always_ask"))
      game = env.Game(self.args)
      game.reset()
      reward = []
      game_reward = []
      lens = []
      interactions = []
      fail = 0
      for i in range(test_num):
         ## eval_policy ##
         if game.seed == None:
            eval_reward, eval_len, eval_interactions, eval_game_reward = game.ask_eval(game.env_fn, seed = i, show_dialogue=game.show_dialogue)
         else:
            eval_reward, eval_len, eval_interactions, eval_game_reward = game.ask_eval(game.env_fn, seed = game.seed, show_dialogue=game.show_dialogue)
         print("task %s, reward %s, len %s, interaction %s, reward w/o comm penalty %s" %(i, eval_reward,eval_len, eval_interactions, eval_game_reward))
         # if eval_reward <= 0:
         if eval_len == 100: 
               fail += 1

         reward.append(eval_reward)
         game_reward.append(eval_game_reward)
         lens.append(eval_len)
         interactions.append(eval_interactions)

      print("Mean reward:", np.mean(reward))
      print("Mean reward w/o comm penalty:", np.mean(game_reward))
      print("Mean len:", np.mean(lens))
      print("Mean interactions:", np.mean(interactions))
      print("Planning success rate:", 1.- fail/test_num)
if __name__ == '__main__':
   pass


