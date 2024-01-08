import argparse
import os,json, sys
import numpy as np
# single gpu    

os.system('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > tmp.txt')
memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu)) 
os.system('rm tmp.txt')

import torch
import utils
print("1")
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("2")


if __name__ == "__main__":
    utils.print_logo(subtitle="Maintained by Research Center for Applied Mathematics and Machine Intelligence, Zhejiang Lab")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SimpleDoorKey", help="SimpleDoorKey, KeyInBox, RandomBoxKey, ColoredDoorKey") 
    parser.add_argument("--save_name", type=str, required=True, help="path to folder containing policy and run details")
    parser.add_argument("--logdir", type=str, default="./log/")          # Where to log diagnostics to
    parser.add_argument("--record", default=False, action='store_true') 
    parser.add_argument("--seed", default=None)
    parser.add_argument("--ask_lambda", type=float, default=0.01, help="weight on communication penalty term")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--n_itr", type=int, default=1000, help="Number of iterations of the learning algorithm")
    parser.add_argument("--policy",   type=str, default='ppo')
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--traj_per_itr", type=int, default=10)
    parser.add_argument("--show", default=False, action='store_true')
    parser.add_argument("--test_num", type=int, default=100)
    
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--run_seed_list", type=int, nargs="*", default=[0])
    print("3")



    if sys.argv[1] == 'eval':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
 
        output_dir = os.path.join(args.logdir, args.policy, args.task, args.save_name)
        
        policy = torch.load(output_dir + "/acmodel.pt")
        policy.eval()
        eval = utils.Eval(args,policy)
        eval.eval_policy(args.test_num)
        print("4")
    elif sys.argv[1] == 'eval_RL':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        output_dir = os.path.join(args.logdir, args.policy, args.task, args.save_name)
   
        policy = torch.load(output_dir + "/acmodel.pt")
        policy.eval()
        eval = utils.Eval(args,policy)
        eval.eval_RL_policy(args.test_num)
        print("5")

    elif sys.argv[1] == 'train':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        from env.Game import Game

        for i in args.run_seed_list:
            setup_seed(i)
            args.save_name = args.save_name + str(i)
            game = Game(args, run_seed=i)
            game.reset()
            game.train()
        print("6")
    elif sys.argv[1] == 'train_RL':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        from env.Game_RL import Game_RL
        game = Game_RL(args)
        game.reset()
        game.train()
        print("7")
    elif sys.argv[1] == 'baseline':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        eval = utils.Eval(args)
        eval.eval_baseline(args.test_num)
        print("8")
    elif sys.argv[1] == 'random':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        eval = utils.Eval(args)
        eval.eval_policy(args.test_num)
        print("9")
    elif sys.argv[1] == 'always':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        eval = utils.Eval(args)
        eval.eval_always_ask(args.test_num)
        print("10")
    else:
        print("Invalid option '{}'".format(sys.argv[1]))
        print("11")

