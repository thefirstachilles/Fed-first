import numpy as np 
import gym 
import torch 
import random
from argparse import ArgumentParser 
import os 
import pandas as pd 

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d


def save(agent, rewards, args): 

    path = './runs/{}/'.format(args.env)
    try: 
        os.makedirs(path)
    except: 
        pass 

    torch.save(agent.q.state_dict(), os.path.join(path, 'model_state_dict'))

    plt.cla()
    plt.plot(rewards, c = 'r', alpha = 0.3)
    plt.plot(gaussian_filter1d(rewards, sigma = 5), c = 'r', label = 'Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns = ['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index = False)







