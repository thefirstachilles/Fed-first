import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from fed_server_env import Fed_server
import json
import argparse
def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-tne', '--train_eps', type=int, default=100,  help='train_eps')
    parser.add_argument('-tte', '--test_eps', type=int, default=10,  help='test_eps')
    parser.add_argument('-ms', '--max_steps', type=int, default=200,  help='max_steps')
    parser.add_argument('-bs', '--batch_size', type=int, default=128,  help='batch_size')
    parser.add_argument('-memo', '--memory_capacity', type=int, default=10000,  help='memory_capacity')
    parser.add_argument('-lra', '--lr_a', type=float, default=0.0001,  help='lr_a')
    parser.add_argument('-lrc', '--lr_c', type=float, default=0.002,  help='lr_c')
    parser.add_argument('-ga', '--gamma', type=float, default=0.95,  help='gamma')
    parser.add_argument('-si', '--sigma', type=float, default=0.01,  help='sigma')
    parser.add_argument('-ta', '--tau', type=float, default=0.005,  help='tau')
    parser.add_argument('-ahd', '--actor_hidden_dim', type=int, default=256, help='actor_hidden_dim')
    parser.add_argument('-chd', '--critic_hidden_dim', type=int, default=256, help='critic_hidden_dim')
    parser.add_argument('-ns', '--n_states', type=int, default=None, help='n_states')
    parser.add_argument('-na', '--n_actions', type=int, default=None, help='n_actions')
    parser.add_argument('-ab', '--action_bound', type=float, default=None, help='action_bound')
    
    
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-dt', '--data', type=str, default='mnist', help='data type')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=40, help='numer of the clients')
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
    parser.add_argument('-pn', '--punish', type=float, default=60, help='punish')
    parser.add_argument('-rm', '--reward_method', type=str, default='constant', help='reward_method')
    
    parser.add_argument('-enf', '--env_frame', type=int, default=30, help='env_frame')
    parser.add_argument('-bw', '--bandwidth', type=float, default=200*10**6, help='bandwidth')
    parser.add_argument('-ds', '--data_size', type=float, default=30*8*10**3, help='data_size')
    parser.add_argument('-bn', '--band_num', type=int, default=2, help='band_num')
    parser.add_argument('-tm', '--t_max', type=float, default=5, help='t_max')
    return parser.parse_args().__dict__

class Config:
    def __init__(self):
        self.args = init_args()
        # self.seed = random.randint(0, 100)
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')



class ReplayBuffer:
    def __init__(self, cfg, path):    
        try:
            with open(path,  "rb") as file:
                self.buffer = np.load(file,allow_pickle=True)
                file.close()
        except:
            self.buffer = np.empty(cfg.args['memory_capacity'], dtype=object)
        self.size = 0
        self.pointer = 0
        self.capacity = cfg.args['memory_capacity']
        self.batch_size = cfg.args['batch_size']
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

    def push(self, transitions):
        self.buffer[self.pointer] = transitions
        self.size = min(self.size + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def clear(self):
        self.buffer = np.empty(self.capacity, dtype=object)
        self.size = 0
        self.pointer = 0

    def sample(self):
        batch_size = min(self.batch_size, self.size)
        indices = np.random.choice(self.size, batch_size, replace=False)
        samples = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
                                             device=self.device), zip(*self.buffer[indices]))
        return samples


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(cfg.args['n_states'], cfg.args['actor_hidden_dim'])
        self.fc2 = nn.Linear(cfg.args['actor_hidden_dim'], cfg.args['actor_hidden_dim'])
        self.fc3 = nn.Linear(cfg.args['actor_hidden_dim'], cfg.args['n_actions'])
        self.action_bound = cfg.args['action_bound']

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))*self.action_bound + self.action_bound
        return action

class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(cfg.args['n_states'] + cfg.args['n_actions'], cfg.args['critic_hidden_dim'])
        self.fc2 = nn.Linear(cfg.args['critic_hidden_dim'], cfg.args['critic_hidden_dim'])
        self.fc3 = nn.Linear(cfg.args['critic_hidden_dim'], 1)


    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class DDPG:
    def __init__(self, cfg, dir_name):
        self.cfg = cfg
        self.memory = ReplayBuffer(cfg, "./res/{}/memory.npy".format(dir_name))
        self.actor = Actor(cfg).to(cfg.device)
        self.actor_target = Actor(cfg).to(cfg.device)
        self.critic = Critic(cfg).to(cfg.device)
        self.critic_target = Critic(cfg).to(cfg.device)
        # self.actor = Actor(cfg).load_state_dict(torch.load("./res/{}/a_model.pt".format(dir_name))).to(cfg.device)
        try: 
            self.actor.load_state_dict(torch.load("./res/{}/a_model.pt".format(dir_name)))
            self.actor_target.load_state_dict(torch.load("./res/{}/at_model.pt".format(dir_name)))
            self.critic.load_state_dict(torch.load("./res/{}/c_model.pt".format(dir_name)))
            self.critic_target.load_state_dict(torch.load("./res/{}/ct_model.pt".format(dir_name)))
        except:
            print('no model')
            
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.args['lr_a'])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.args['lr_c'])
        self.critic_target.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.cfg.device)
        a = self.actor(state).squeeze(0).cpu().numpy()
        a += self.cfg.args['sigma'] * np.random.randn(self.cfg.args['n_actions'])
        
        return a


    def update(self):
        if self.memory.size < self.cfg.args['batch_size']:
            return 0, 0
        states, actions, rewards, next_states, dones = self.memory.sample()
        rewards, dones =  rewards.view(-1, 1), dones.view(-1, 1)
        next_q_value = self.critic_target(next_states.squeeze(1), self.actor_target(next_states).squeeze(1))
        target_q_value = rewards + (1 - dones) * self.cfg.args['gamma'] * next_q_value

        critic_loss = torch.mean(F.mse_loss(self.critic(states.squeeze(1), actions.squeeze(1)), target_q_value))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -torch.mean(self.critic(states.squeeze(1), self.actor(states).squeeze(1)))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.update_params()

        return actor_loss.item(), critic_loss.item()

    def update_params(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.cfg.args['tau'] * param.data +
                                    (1. - self.cfg.args['tau']) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.cfg.args['tau'] * param.data +
                                    (1. - self.cfg.args['tau']) * target_param.data)


def env_agent_config(cfg):
    rng2 = np.random.default_rng(seed=100)
    prop_path = 'lr_a_{}_lr_c_{}'.format(cfg.args['lr_a'],cfg.args['lr_c'])
    env = Fed_server(rng2, cfg.args, prop_path, 'DDPG')
    cfg.args['n_states'] = 2*env.args['num_of_clients']
    # cfg.n_actions = env.action_space.shape[0]
    cfg.args['n_actions'] = env.args['num_of_clients']
    cfg.args['action_bound'] = 0.5
    dir_name = env.get_path()
    agent = DDPG(cfg,dir_name)
    return env, agent


def train(env, agent, cfg):
    rewards, steps = [], []
    for i in range(cfg.args['train_eps']):
        ep_reward, ep_step = 0.0, 0
        state = env.get_env_info(0)
        critic_loss, actor_loss = 0.0, 0.0
        for round in range(cfg.args['env_frame']):
            ep_step += 1
            a = agent.choose_action(state)
            action = np.zeros(env.args['num_of_clients'], dtype=int)
            for index, ele in enumerate(a):
                if ele > 0.5:
                    action[index] = 1
            done, reward = env.train_process(action, round)
            next_state = env.get_env_info(round+1)
            agent.memory.push((state, a, reward, next_state, done))
            state = next_state
            c_loss, a_loss = agent.update()
            critic_loss += c_loss
            actor_loss += a_loss
            ep_reward += reward
            if done:
                env.reset()
                path = env.get_path()
                torch.save(agent.actor.state_dict(), "./res/{}/a_model.pt".format(path))
                torch.save(agent.actor_target.state_dict(), "./res/{}/at_model.pt".format(path))
                torch.save(agent.critic.state_dict(), "./res/{}/c_model.pt".format(path))
                torch.save(agent.critic_target.state_dict(), "./res/{}/ct_model.pt".format(path))
                with open("./res/{}/memory.npy".format(path),  "wb") as file:
                    np.save(file, agent.memory.buffer)
                    # file.write(json.dumps(agent.memory.buffer))
                    file.close() 
                break
            
    return rewards, steps


# def test(agent, cfg):
#     print('开始测试!')
#     rewards, steps = [], []
#     env = gym.make(cfg.env_name, render_mode='human')
#     for i in range(cfg.test_eps):
#         ep_reward, ep_step = 0.0, 0
#         state, _ = env.reset(seed=cfg.seed)
#         for _ in range(cfg.max_steps):
#             ep_step += 1
#             a = agent.choose_action(state)
#             action = []
#             for index, ele in enumerate(a):
#                 if ele > 0.5:
#                     action.append(index)
#             done, reward = env.train_process(action)
#             state = env.get_env_info()
#             ep_reward += reward
#             if terminated or truncated:
#                 break
#         steps.append(ep_step)
#         rewards.append(ep_reward)
#         print(f'回合:{i + 1}/{cfg.test_eps}, 奖励:{ep_reward:.3f}')
#     print('结束测试!')
#     env.close()
#     return rewards, steps


if __name__ == '__main__':
    cfg = Config()
    env, agent = env_agent_config(cfg)
    train_rewards, train_steps = train(env, agent, cfg)
    # test_rewards, test_steps = test(agent, cfg)