from collections import OrderedDict
import os, pickle
import json

class color:
 BOLD   = '\033[1m\033[48m'
 END    = '\033[0m'
 ORANGE = '\033[38;5;202m'
 BLACK  = '\033[38;5;240m'


def create_logger(args):
    from torch.utils.tensorboard import SummaryWriter
    """Use hyperparms to set a directory to output diagnostic files."""

    arg_dict = args.__dict__

    assert "logdir" in arg_dict, \
      "You must provide a 'logdir' key in your command line arguments."
  
    arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))
    logdir = str(arg_dict.pop('logdir'))
    output_dir = os.path.join(logdir, args.policy, args.task, args.save_name)

    os.makedirs(output_dir, exist_ok=True)

    # Create a file with all the hyperparam settings in plaintext
    info_path = os.path.join(output_dir, "config.json")

    with open(info_path,'wt') as f:
        json.dump(arg_dict, f, indent=4)

    logger = SummaryWriter(output_dir, flush_secs=0.1)
    print("Logging to " + color.BOLD + color.ORANGE + str(output_dir) + color.END)

    logger.name = args.save_name
    logger.dir = output_dir
    return logger

def parse_previous(args):
    if args.previous is not None:
        run_args = pickle.load(open(args.previous + "config.json", "rb"))
        args.recurrent = run_args.recurrent
        args.env_name = run_args.env_name
        args.command_profile = run_args.command_profile
        args.input_profile = run_args.input_profile
        args.learn_gains = run_args.learn_gains
        args.traj = run_args.traj
        args.no_delta = run_args.no_delta
        args.ik_baseline = run_args.ik_baseline
        if args.exchange_reward is not None:
            args.reward = args.exchange_reward
            args.run_name = run_args.run_name + "_NEW-" + args.reward
        else:
            args.reward = run_args.reward
            args.run_name = run_args.run_name + "--cont"
    return args
