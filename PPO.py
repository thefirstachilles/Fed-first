#PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import argparse
import numpy as np
from fed_server_env import Fed_server 

#Hyperparameters
def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    ## bqn 参数
    # parser.add_argument('-es', '--epsilon_start', type=float, default=1,  help='epsilon start')
    # parser.add_argument('-ef', '--epsilon_final', type=float, default=0.01,  help='epsilon final')
    # parser.add_argument('-ed', '--epsilon_decay', type=int, default=8000,  help='epsilon decay')
    # parser.add_argument('-ms', '--memory_size', type=int, default=100000,  help='memory_size')
    parser.add_argument('-tne', '--train_eps', type=int, default=100,  help='train_eps')
    parser.add_argument('-lpl', '--lstm_ppo_lr', type=float, default=0.001,  help='learning_rate')
    parser.add_argument('-ga', '--gamma', type=float, default=0.99,  help='gamma')
    parser.add_argument('-lmbda', '--lmbda', type=float, default=0.95,  help='lambda')
    parser.add_argument('-ec', '--eps_clip', type=float, default=0.001,  help='eps_clip')
    parser.add_argument('-ke', '--k_epoch', type=int, default=2,  help='k_epoch')
    parser.add_argument('-Th', '--T_horizon', type=int, default=50,  help='T_horizon')
    
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

class PPO(nn.Module):
    def __init__(self, obs, ac, n, args):
        super().__init__()
        self.args = args
        self.data = []
        self.fc1   = nn.Linear(obs,64)
        self.lstm  = nn.LSTM(64,32)
        self.fc_pi = nn.ModuleList([nn.Linear(32, n) for i in range(ac)])
        self.fc_v  = nn.Linear(32,1)
        # self.softmax = nn.Softmax

        self.optimizer = optim.Adam(self.parameters(), lr=self.args['lstm_ppo_lr'])

        self.gamma = args['gamma']
        self.lmbda = args['lmbda']
        self.eps_clip = args['eps_clip']
       
    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        # x = torch.stack([l(x) for l in self.fc_pi], dim = 1)
        # prob = F.softmax(x, dim=3)
        prob = torch.stack([F.softmax(l(x),dim = 2) for l in self.fc_pi], dim = 1)

        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.int64), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(self.args['k_epoch']):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            # pi_a = prob.gather(1,torch.tensor(action.reshape(10,1)))
            pi_a = pi.squeeze(2).gather(2,a.unsqueeze(2))
            # pi_a = []
            # for i, action_branch in enumerate(pi):
            #     pi_a.append(action_branch.probs.squeeze(1).gather(1, a[:,i].unsqueeze(1)))
            # pi_a = torch.stack(pi_a, dim=2).squeeze(1)
            ratio = torch.exp((torch.log(pi_a.squeeze(2))- torch.log(prob_a)).sum(1).unsqueeze(1))# a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())
            print('loss',loss)
            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
        
def main():
    rng2 = np.random.default_rng(seed=100)
    args = init_args()
    prop_path = 'ppoLr_{}_kEpoch_{}'.format(args['lstm_ppo_lr'],args['k_epoch'])
    env = Fed_server(rng2, args, prop_path, 'PPO')
    bins = 2
    model = PPO(2*env.args['num_of_clients'], env.args['num_of_clients'], bins, args)
    dir_name = env.get_path()
    try: 
        model.load_state_dict(torch.load("./res/{}/model.pt".format(dir_name)))
        print('has model')
    except:
        print('no model')
    print_interval = 20
    recap = []
    
    for n_epi in range(args['train_eps']):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        s = env.get_env_info(0)
        done = False
        
        while not done:
            for t in range(args['T_horizon']):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(env.args['num_of_clients'],-1)
                action = np.ones( env.args['num_of_clients'], dtype=int)
                for index, prob_item in enumerate(prob[:]):                 
                    m = Categorical(prob_item)
                    action[index] = m.sample().item()
                # prob_a = prob.gather(3,a.view(1,-1,1,1))
                # action = a.detach().numpy().reshape(-1)
                # _actions = []
                # probs = []
                # for action_branch in policy:
                #     action = action_branch.sample()
                #     _actions.append(action.item())
                #     probs.append(action_branch.probs[0,0,action.item()].item())
                
                done, r = env.train_process(action,t)
                s_prime = env.get_env_info(t+1)

                # s_prime, r, done, = env.get_env_info(a)
                
                model.put_data((s, action, r, s_prime, prob.gather(1,torch.tensor(action.reshape(env.args['num_of_clients'],1))).view(-1).detach().numpy(), h_in, h_out, done))
                s = s_prime

                recap.append(r)
                # with open('test-branch-lstm-ppo.txt','w') as file:
                #     file.write(str(recap))
                #     file.close()

                if done:
                    path = env.get_path()
                    torch.save(model.state_dict(), "./res/{}/model.pt".format(path))
                    env.reset()
                    break
                    
            model.train_net()

        # if n_epi%print_interval==0 and n_epi!=0:
        #     print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        #     score = 0.0


if __name__ == '__main__':
    main()