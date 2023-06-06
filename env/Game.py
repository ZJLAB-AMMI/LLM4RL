#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Game.py
@Time    :   2023/05/25 11:06:59
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
from planner import Planner
from commuication_net import Communication_Net
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

prefix = os.getcwd()
task_info_json = os.path.join(prefix, "prompt/task_info.json")
class Game:
    def __init__(self, args, policy=None):
         
        self.task = args.task
        self.device = args.device
        self.planner = Planner(self.task) 
        self.max_ep_len, self.decription, self.task_level, self.task_example, self.configurations = self.load_task_info(self.task)
        if args.seed is not None:
            self.seed = int(args.seed)
        else:
            self.seed = args.seed
        self.ask_lambda = args.ask_lambda
        self.gamma = args.gamma
        self.lam = args.lam
        self.batch_size = args.batch_size
        self.frame_stack = args.frame_stack

        self.env_fn = utils.make_env_fn(self.configurations, render_mode="rgb_array", frame_stack = args.frame_stack)
        self.agent_view_size = self.env_fn().agent_view_size

        obs_space, preprocess_obss = utils.get_obss_preprocessor(self.env_fn().observation_space)
        self.record_frames = args.record
        if policy is None:
            ## communication network for ask
            self.Communication_net = Communication_Net(obs_space, 2).to(self.device)
        else:
            self.Communication_net = policy.to(self.device)
        self.buffer = algos.Buffer(self.gamma, self.lam, self.device)
        
        self.logger = utils.create_logger(args)
    
        self.ppo_algo = algos.PPO(self.Communication_net, device=self.device, save_path=self.logger.dir, batch_size=self.batch_size)

        self.n_itr = args.n_itr
        self.total_steps = 0
        self.show_dialogue = args.show
 
        ## global_param for explore skill
        utils.global_param.init()
        self.traj_per_itr = args.traj_per_itr

    def load_task_info(self, task):
        with open(task_info_json, 'r') as f:
            task_info = json.load(f)
        episode_length = int(task_info[task]["episode"])
        task_description = task_info[task]['description']
        task_example = task_info[task]['example']
        task_level = task_info[task]['level']
        task_configurations = task_info[task]['configurations']
        return episode_length, task_description, task_level, task_example, task_configurations       
    

    def reset(self):
        print(f"[INFO]: resetting the task: {self.task}")
        self.planner.initial_planning(self.decription, self.task_example)

    def train(self):
        start_time = time.time()
        for itr in range(self.n_itr):
            print("********** Iteration {} ************".format(itr))
            print("time elapsed: {:.2f} s".format(time.time() - start_time))

            ## collecting ##
            sample_start = time.time()
            buffer = []
            for _ in range(self.traj_per_itr):
                buffer.append(self.collect(self.env_fn,seed=self.seed))
            self.buffer = algos.Merge_Buffers(buffer,device=self.device)
            total_steps = len(self.buffer)
            samp_time = time.time() - sample_start
            print("{:.2f} s to collect {:6n} timesteps | {:3.2f}sample/s.".format(samp_time, total_steps, (total_steps)/samp_time))
            self.total_steps += total_steps

            ## training ##
            optimizer_start = time.time()
            mean_losses = self.ppo_algo.update_policy(self.buffer)
            opt_time = time.time() - optimizer_start
            print("{:.2f} s to optimizer| loss {:6.3f}, entropy {:6.3f}.".format(opt_time, mean_losses[0], mean_losses[1]))

            ## eval_policy ##
            evaluate_start = time.time()
            eval_reward, eval_len, eval_interactions, eval_game_reward = self.eval(self.env_fn, trajs=1, seed=self.seed, show_dialogue=self.show_dialogue)
            eval_time = time.time() - evaluate_start
            print("{:.2f} s to evaluate.".format(eval_time))
            if self.logger is not None:
                avg_eval_reward = eval_reward
                avg_batch_reward = np.mean(self.buffer.ep_returns)
                std_batch_reward = np.std(self.buffer.ep_returns)
                avg_batch_game_reward = np.mean(self.buffer.ep_game_returns)
                std_batch_game_reward = np.std(self.buffer.ep_game_returns)
                avg_ep_len = np.mean(self.buffer.ep_lens)
                success_rate = sum(i<100 for i in self.buffer.ep_lens) / 10.
                avg_eval_len = eval_len
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Timesteps', self.total_steps) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (test)', round(avg_eval_reward,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Ep Lens (test) ', round(avg_eval_len,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Ep Comm (test) ', round(eval_interactions,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (batch)', round(avg_batch_reward,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', round(avg_ep_len,2)) + "\n")
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()

                self.logger.add_scalar("Test/Return", avg_eval_reward, itr)
                self.logger.add_scalar("Test/Game Return", eval_game_reward, itr)
                self.logger.add_scalar("Test/Mean Eplen", avg_eval_len, itr)
                self.logger.add_scalar("Test/Comm", eval_interactions, itr)
                self.logger.add_scalar("Train/Return Mean", avg_batch_reward, itr)
                self.logger.add_scalar("Train/Return Std", std_batch_reward, itr)
                self.logger.add_scalar("Train/Game Return Mean", avg_batch_game_reward, itr)
                self.logger.add_scalar("Train/Game Return Std", std_batch_game_reward, itr)
                self.logger.add_scalar("Train/Eplen", avg_ep_len, itr)
                self.logger.add_scalar("Train/Success Rate", success_rate, itr)
                self.logger.add_scalar("Train/Loss", mean_losses[0], itr)
                self.logger.add_scalar("Train/Mean Entropy", mean_losses[1], itr)

                self.ppo_algo.save()


    def collect(self, env_fn, seed=None):
        # Do one agent-environment interaction
        env = utils.WrapEnv(env_fn)
        with torch.no_grad():
            buffer = algos.Buffer(self.gamma, self.lam)
            obs = env.reset(seed)
            self.planner.reset()

            done = False
            skill_done = True
            traj_len = 0
            interactions = 0
            pre_skill = None
            if self.frame_stack >1:
                com_obs = obs
            else:
                his_obs = obs
                com_obs = obs - his_obs

            utils.global_param.set_value('exp', None)
            utils.global_param.set_value('explore_done', False)
            while not done and traj_len < self.max_ep_len:
                dist, value = self.Communication_net(torch.Tensor(com_obs).to(self.device))         
                ask_flag = dist.sample()
                log_probs = dist.log_prob(ask_flag)
                if ask_flag == 1:
                    interactions += 1
                if skill_done or ask_flag:
                    skill = self.planner(obs)
                    # print(skill)
                    if pre_skill == skill: # additional penalty term for repeat same skill
                        repeat_feedback = np.array([1.])
                    elif pre_skill is None:
                        repeat_feedback = np.array([0.])
                    else:
                        repeat_feedback = np.array([-0.1])
                    pre_skill = skill
                    self.Executive_net = Executive_net(skill,obs[0],self.agent_view_size)

                ## RL choose skill, and return action_list
                action, skill_done = self.Executive_net(obs[0])

                ## one step do one action in action_list
                next_obs, reward, done, info = env.step(np.array([action]))

                comm_penalty = (self.ask_lambda + 0.1 * repeat_feedback) * ask_flag.to("cpu").numpy() ## communication penalty
                comm_reward = reward - comm_penalty 
    
                buffer.store(com_obs, ask_flag.to("cpu").numpy(), comm_reward, value.to("cpu").numpy(), log_probs.to("cpu").numpy(), reward) #TODO:check shape
                if self.frame_stack >1:
                    obs = next_obs
                    com_obs = obs
                else:
                    his_obs = obs
                    obs = next_obs
                    com_obs = obs - his_obs
                traj_len += 1
            _, value = self.Communication_net(torch.Tensor(com_obs).to(self.device))
            buffer.finish_path(last_val=(not done) * value.to("cpu").numpy(), interactions=interactions)

        return buffer




    def eval(self, env_fn, trajs=1, seed=None, show_dialogue=False):
        env = utils.WrapEnv(env_fn)
        with torch.no_grad():
            ep_returns = []
            ep_game_returns = []
            ep_lens = []
            ep_interactions = []
            for traj in range(trajs):
                obs = env.reset(seed)
                #print(f"[INFO]: Evaluating the task is ", self.task)
                self.planner.reset(show_dialogue)
                
                if self.record_frames:
                    video_frames = [obs['rgb']]
                    goal_frames = ["start"] 

                done = False
                skill_done = True
                traj_len = 0
                pre_skill = None
                ep_return = 0
                ep_game_return = 0
                interactions = 0
                if self.frame_stack > 1:
                    com_obs = obs
                else:
                    his_obs = obs
                    com_obs = obs - his_obs
                utils.global_param.set_value('exp', None)
                utils.global_param.set_value('explore_done', False)
                while not done and traj_len < self.max_ep_len:
                    dist, _ = self.Communication_net(torch.Tensor(com_obs).to(self.device))
                    ask_flag = dist.sample()
                    if ask_flag == 1:
                        interactions += 1
                    if skill_done or ask_flag:
                        skill = self.planner(obs)
                        # print(skill)
                        if pre_skill == skill: # additional penalty term for repeat same skill
                            repeat_feedback = np.array([1.])
                        elif pre_skill is None:
                            repeat_feedback = np.array([0.])
                        else:
                            repeat_feedback = np.array([-0.1])
                        pre_skill = skill
                        self.Executive_net = Executive_net(skill,obs[0],self.agent_view_size)

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
                    comm_penalty = (self.ask_lambda + 0.1 * repeat_feedback) * ask_flag.to("cpu").numpy()  ## communication penalty
                    comm_reward = reward - comm_penalty
                    #reward = 1.0*reward
                    ep_return += comm_reward
                    ep_game_return += 1.0*reward
                    traj_len += 1
                if show_dialogue:
                    if done:
                        print("RL: Task Completed! \n")
                    else:
                        print("RL: Task Fail! \n")
                ep_returns += [ep_return]
                ep_game_returns += [ep_game_return]
                ep_lens += [traj_len]
                ep_interactions += [interactions]
            return np.mean(ep_returns), np.mean(ep_lens), np.mean(ep_interactions), np.mean(ep_game_returns)

    def baseline_eval(self, env_fn, trajs=1, seed=None, show_dialogue=False):
        env = utils.WrapEnv(env_fn)
        with torch.no_grad():
            ep_returns = []
            ep_game_returns = []
            ep_lens = []
            ep_interactions = []
            for traj in range(trajs):
                obs = env.reset(seed)
                # print(f"[INFO]: Evaluating the task is ", self.task)
                self.planner.reset(show_dialogue)
                
                if self.record_frames:
                    video_frames = [obs['rgb']]
                    goal_frames = ["start"] 

                done = False
                skill_done = True
                traj_len = 0
                ep_return = 0
                ep_game_return = 0
                interactions = 0
                utils.global_param.set_value('exp', None)
                utils.global_param.set_value('explore_done', False)
                while not done and traj_len < self.max_ep_len:
                    if skill_done:
                        skill = self.planner(obs)
                        # print(skill)
                        self.Executive_net = Executive_net(skill,obs[0],self.agent_view_size)
                        interactions += 1
                    ## RL choose skill, and return action_list
                    action, skill_done = self.Executive_net(obs[0])
                    ## one step do one action in action_list
                    obs, reward, done, info = env.step(np.array([action]))
                    comm_reward = reward - self.ask_lambda * float(skill_done) ## communication penalty
                
                    ep_return += comm_reward
                    ep_game_return += 1.0*reward
                    traj_len += 1
                if show_dialogue:
                    if done:
                        print("RL: Task Completed! \n")
                    else:
                        print("RL: Task Fail! \n")
                ep_returns += [ep_return]
                ep_game_returns += [ep_game_return]
                ep_lens += [traj_len]
                ep_interactions += [interactions]
            return np.mean(ep_returns), np.mean(ep_lens), np.mean(ep_interactions), np.mean(ep_game_returns)
        
    def ask_eval(self, env_fn, trajs=1, seed=None, show_dialogue=False):
        env = utils.WrapEnv(env_fn)
        with torch.no_grad():
            ep_returns = []
            ep_game_returns = []
            ep_lens = []
            ep_interactions = []
            for traj in range(trajs):
                obs = env.reset(seed)
                #print(f"[INFO]: Evaluating the task is ", self.task)
                self.planner.reset(show_dialogue)
                if self.record_frames:
                    video_frames = [obs['rgb']]
                    goal_frames = ["start"] 

                done = False
                skill_done = True
                traj_len = 0
                pre_skill = None
                ep_return = 0
                ep_game_return = 0
                utils.global_param.set_value('exp', None)
                utils.global_param.set_value('explore_done', False)
                while not done and traj_len < self.max_ep_len:
                    ## always ask
                    ask_flag = torch.Tensor([1])
                    if skill_done or ask_flag:
                        skill = self.planner(obs)
                        # print(skill)
                        if pre_skill == skill: # additional penalty term for repeat same skill
                            repeat_feedback = np.array([1.])
                        elif pre_skill is None:
                            repeat_feedback = np.array([0.])
                        else:
                            repeat_feedback = np.array([-0.1])
                        pre_skill = skill
                        self.Executive_net = Executive_net(skill,obs[0],self.agent_view_size)

                    ## RL choose skill, and return action_list
                    action, skill_done = self.Executive_net(obs[0])
                    ## one step do one action in action_list
                    obs, reward, done, info = env.step(np.array([action]))
                    comm_penalty = (self.ask_lambda + 0.1 * repeat_feedback) * ask_flag.to("cpu").numpy()  ## communication penalty
                    comm_reward = reward - comm_penalty 
                    #reward = 1.0*reward
                    ep_return += comm_reward
                    ep_game_return += 1.0*reward
                    traj_len += 1
                if show_dialogue:
                    if done:
                        print("RL: Task Completed! \n")
                    else:
                        print("RL: Task Fail! \n")
                ep_returns += [ep_return]
                ep_game_returns += [ep_game_return]
                ep_lens += [traj_len]
                ep_interactions += [traj_len]
            return np.mean(ep_returns), np.mean(ep_lens), np.mean(ep_interactions), np.mean(ep_game_returns)
if __name__ == '__main__':
   pass

