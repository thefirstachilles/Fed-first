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
    
    
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-dt', '--data', type=str, default='mnist', help='data type')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=20, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.2, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
    parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=2000, help='number of communications')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=bool, default=False, help='the way to allocate data to clients')
    parser.add_argument('-isc', '--iscontinue', type=int, default=0, help='if use some model continue')
    parser.add_argument('-moment', '--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-ib', '--isblur', type=str, default=False, help='if blur the dataset')
    parser.add_argument('-sc', '--sample_client', type=int, default=10, help='sample_client')
    parser.add_argument('-rw', '--reward_weight', type=float, default=0.5, help='reward_weight')
    parser.add_argument('-ss', '--set_size', type=int, default=100, help='set_size')
    parser.add_argument('-pn', '--punish', type=float, default=30, help='punish')
    parser.add_argument('-rm', '--reward_method', type=str, default='variable', help='reward_method')
    parser.add_argument('-sen', '--select_num', type=str, default='constant', help='select_num')
    
    parser.add_argument('-enf', '--env_frame', type=int, default=30, help='env_frame')
    parser.add_argument('-bw', '--bandwidth', type=float, default=200*10**6, help='bandwidth')
    parser.add_argument('-ds', '--data_size', type=float, default=30*8*10**3, help='data_size')
    parser.add_argument('-bn', '--band_num', type=int, default=2, help='band_num')
    parser.add_argument('-tm', '--t_max', type=float, default=5, help='t_max')
    return parser.parse_args().__dict__
if __name__=="__main__":
    args = init_args()
    rng2 = np.random.default_rng(seed=100)
    prop_path = ''
    env = Fed_server(rng2, args, prop_path,'Random')
    for frame in range(args['max_ep']): 
        s = env.get_env_info(frame)
        action = rng2.integers(0,2,env.args['num_of_clients'])
        done,r = env.train_process(action, frame)
        ns = env.get_env_info(frame+1)
        if done:
            env.reset()
            