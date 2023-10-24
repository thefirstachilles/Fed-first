import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

import numpy as np 

import random 
import argparse
from fed_server_env import Fed_server 
def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-ep', '--max_ep', type=int, default=3000,  help='max_frames')
    return parser.parse_args().__dict__
if __name__=="__main__":
    args = init_args()
    rng2 = np.random.default_rng(seed=100)
    env = Fed_server(rng2, 'Random')
    for frame in range(args['max_ep']): 
        s = env.get_env_info()
        action = rng2.integers(0,2,env.args['num_of_clients'])
        done,r = env.train_process(action, frame)
        ns = env.get_env_info()
        if done:
            env.reset()
            