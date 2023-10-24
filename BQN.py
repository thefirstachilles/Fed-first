from tqdm import tqdm
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

import numpy as np 

import random 
import argparse
from branch_model import DuelingNetwork, BranchingQNetwork
from fed_server_env import Fed_server 

def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    ## bqn 参数
    parser.add_argument('-es', '--epsilon_start', type=float, default=1.0,  help='epsilon start')
    parser.add_argument('-ef', '--epsilon_final', type=float, default=0.1,  help='epsilon final')
    parser.add_argument('-ed', '--epsilon_decay', type=int, default=1000,  help='epsilon decay')
    parser.add_argument('-ga', '--gamma', type=float, default=0.99,  help='gamma')
    parser.add_argument('-bl', '--bqn_lr', type=float, default=0.001,  help='bqn lr')
    parser.add_argument('-tnuf', '--target_net_update_freq', type=int, default=1000,  help='target net update freq')
    parser.add_argument('-ms', '--memory_size', type=int, default=10000,  help='memory_size')
    parser.add_argument('-bs', '--batch_size', type=int, default=30,  help='batch_size')
    parser.add_argument('-ls', '--learning_starts', type=float, default=60,  help='learning_starts')
    parser.add_argument('-mf', '--max_frames', type=float, default=3000,  help='max_frames')
    return parser.parse_args().__dict__

class BranchingDQN(nn.Module): 

    def __init__(self, obs, ac, n, args): 

        super().__init__()

        self.q = BranchingQNetwork(obs, ac, n)
        self.target = BranchingQNetwork(obs, ac, n)

        self.target.load_state_dict(self.q.state_dict())

        self.target_net_update_freq = args['target_net_update_freq']
        self.update_counter = 0

    def get_action(self, x): 

        with torch.no_grad(): 
            # a = self.q(x).max(1)[1]
            out = self.q(x).squeeze(0)
            action = torch.argmax(out, dim = 1)
            # action_ = np.where(choose == 1)[0]
            # np.random.shuffle(action_)
            # action = action_[:2]
            # action = np.argpartition(act.numpy(), -join_num)[-join_num:]
        return action.numpy()

    def update_policy(self, adam, memory, params): 

        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(params['batch_size'])

        states = torch.tensor(b_states).float()
        actions = torch.tensor(b_actions).long().reshape(states.shape[0],-1,1)
        rewards = torch.tensor(b_rewards).float().reshape(-1,1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1,1)

        qvals = self.q(states)

        current_q_values = self.q(states).gather(2, actions).squeeze(-1)

        with torch.no_grad():
            argmax = torch.argmax(self.q(next_states), dim = 2)
            max_next_q_vals = self.target(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            max_next_q_vals = max_next_q_vals.mean(1, keepdim = True)
        expected_q_vals = rewards + max_next_q_vals*0.99*masks
        # print(expected_q_vals[:5])
        loss = F.mse_loss(expected_q_vals, current_q_values)
        print('loss',loss)
        # input(loss)

        # print('\n'*5)
        
        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters(): 
            p.grad.data.clamp_(-1.,1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0: 
            self.update_counter = 0 
            self.target.load_state_dict(self.q.state_dict())


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])


        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
    


if __name__=="__main__":
    rng2 = np.random.default_rng(seed=100)
    args = init_args()
    bins = 2
    env = Fed_server(rng2, 'BQN')
    memory = ExperienceReplayMemory(args['memory_size'])
    agent = BranchingDQN(2*env.args['num_of_clients'], env.args['num_of_clients'], bins, args)
    adam = optim.Adam(agent.q.parameters(), lr = args['bqn_lr']) 


    s = env.get_env_info()
    ep_reward = 0. 
    recap = []
    i = 0
    for frame in range(args['max_frames']): 
        # if_memo = False
        epsilon = args['epsilon_final']+ (args['epsilon_start'] - args['epsilon_final']) * np.exp(-1. * frame / args['epsilon_decay'])

        if np.random.random() > epsilon: 
            action = agent.get_action(torch.tensor(s, dtype=torch.float32))
            
        else: 
            # if_memo = True
            # 希望在h最好的前提下去随机选择20到40个
            random_num = int(rng2.random()*7)+3
            action = np.zeros(shape = env.args['num_of_clients'], dtype=int)
            candi_action_indices =  np.argsort(env.Sat_to_iot.h_k_los_value)[:10].tolist()
            # 从40个里随机抽取了20-40个设备
            random_indices = rng2.choice(10, random_num, replace=False)
            # 删去candi_action_indices 中没有random上的
            _candi_action_indices = []
            for index, item in enumerate(random_indices):
                _candi_action_indices.append(candi_action_indices[item])
            for index,item in enumerate(_candi_action_indices):
                action[item] = 1
            # action = np.random.randint(2, size=env.args['num_of_clients'])
            # action = np.zeros(shape = env.args['num_of_clients'], dtype=int)
            # random_num = rng2.random()
            # random_sort = rng2.choice(env.args['num_of_clients'], int(env.args['num_of_clients']*random_num/2), replace=False)
            # for index, item in enumerate(random_sort):
            #     action[item] = 1
                
            # a_ = np.zeros(env.args['num_of_clients'])
            # join_num = int(env.args['num_of_clients']*env.args['cfraction'])
            # for i in list(range(join_num)):
            #     a_[i] = 1
            # np.random.shuffle(a_)
            # action = np.random.shuffle(np.hstack((np.ones(int(env.args['num_of_clients']*env.args['cfraction'])), np.zeros(int(env.args['num_of_clients']*(1-env.args['cfraction']))))))
        done, r = env.train_process(action, i)
        ns = env.get_env_info()
        recap.append(r)  
        # with open('test-bqn.txt','w') as file:
        #         file.write(str(recap))
        #         file.close()
        i+=1
        if done:
            env.reset()
            ns = env.get_env_info()
            recap.append(ep_reward)
            ep_reward = 0.  
            i=0
        # if if_memo:
        memory.push((s.reshape(-1).tolist(), action, r, ns.reshape(-1).tolist(), 0. if done else 1.))
        s = ns  

        if frame > args['learning_starts']:
            agent.update_policy(adam, memory, args)



