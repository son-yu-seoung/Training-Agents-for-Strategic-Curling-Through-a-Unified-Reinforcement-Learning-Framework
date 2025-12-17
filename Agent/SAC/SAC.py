import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Agent.SAC.actor import *
from Agent.SAC.critic import *
 
import utils
import pickle

# SAC 주요 기술 적용:
# - CNN을 활용한 상태 인코딩 (컬링 환경에서 state가 이미지이므로 CNN 사용)
# - Twin Q-Network 사용 (Q-value 과대평가 방지)
# - Stochastic Policy (확률적 정책 사용)
# - Entropy Regularization (탐색 강화)

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity):
        self.capacity = capacity

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32) 

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done): 
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs]).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda().float()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda().float()
        next_obses = torch.as_tensor(self.next_obses[idxs]).cuda().float()
        dones = torch.as_tensor(self.dones[idxs]).cuda().float()

        return obses, actions, rewards, next_obses, dones
    

class SACAgent():
    def __init__(self, opt): 
        self.opt = opt 
 
        self.actor = DiagGaussianActor(opt).cuda()
        self.actor_partner = DiagGaussianActor(opt).cuda()
        self.actor_partner.load_state_dict(self.actor.state_dict())

        self.critic = DoubleQCritic(opt).cuda()
        self.critic_target = DoubleQCritic(opt).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(opt.init_temperature)).cuda()
        self.log_alpha.requires_grad = True

        self.target_entropy = - opt.action_dim

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=opt.lr,
                                                betas=opt.beta)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=opt.lr,
                                                 betas=opt.beta)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=opt.lr,
                                                    betas=opt.beta)
        
        self.actor.train()
        self.actor_partner.eval()
        self.critic.train() 


    def network_init(self, eval=False):
        if self.opt.phase == 1:
            if self.opt.s_episode != 0:
                print('network init')
                self.actor.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_{self.opt.s_episode}.pth'))
                self.actor_partner.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_partner_{self.opt.s_episode}.pth')) 

                self.critic.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_{self.opt.s_episode}.pth')) 
                self.critic_target.load_state_dict(self.critic.state_dict())
                
                self.log_alpha = torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/log_alpha_{self.opt.s_episode}.pth')
                self.log_alpha.requires_grad = True 

                self.actor_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_optim_{self.opt.s_episode}.pth'))
                self.critic_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_optim_{self.opt.s_episode}.pth'))
                self.log_alpha_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/log_alpha_optim_{self.opt.s_episode}.pth'))
                

                self.avg_reward_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/avg_reward_list_{self.opt.s_episode}.npy')
                self.actor_loss_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_loss_list_{self.opt.s_episode}.npy')
                self.critic_loss_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_loss_list_{self.opt.s_episode}.npy')
                self.alpha_loss_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/alpha_loss_list_{self.opt.s_episode}.npy')
                self.q_value_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/q_value_list_{self.opt.s_episode}.npy')
                self.avg_return_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/avg_return_list_{self.opt.s_episode}.npy')

            else:
                self.avg_reward_list = np.array([]) 
                self.actor_loss_list = np.array([]) 
                self.critic_loss_list = np.array([]) 
                self.alpha_loss_list = np.array([])  
                self.q_value_list = np.array([])  
                self.avg_return_list = np.array([])
        
        if self.opt.phase == 2:
            if eval:
                self.actor.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_{self.opt.s_episode}.pth'))
                self.actor_partner.load_state_dict(self.actor.state_dict())  
                self.critic.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_{self.opt.s_episode}.pth')) 
                self.critic_target.load_state_dict(self.critic.state_dict())
                   
            else:
                self.actor.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/actor_{self.opt.phase1_episode}.pth'))
                self.actor_partner.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/actor_partner_{self.opt.phase1_episode}.pth')) 

                # self.critic.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/critic_{self.opt.phase1_episode}.pth')) 
                # self.critic_target.load_state_dict(self.critic.state_dict())
                
                # self.log_alpha = torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/log_alpha_{self.opt.phase1_episode}.pth')
                # self.log_alpha.requires_grad = True 

                # self.actor_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/actor_optim_{self.opt.phase1_episode}.pth'))
                # self.critic_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/critic_optim_{self.opt.phase1_episode}.pth'))
                # self.log_alpha_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/log_alpha_optim_{self.opt.phase1_episode}.pth'))
            
                self.avg_reward_list = np.array([]) 
                self.actor_loss_list = np.array([]) 
                self.critic_loss_list = np.array([]) 
                self.alpha_loss_list = np.array([])  
                self.q_value_list = np.array([])  
                self.avg_return_list = np.array([])


    def log_episode_return(self, episode_return):
        self.avg_return_list = np.append(self.avg_return_list, episode_return)


    @property
    def alpha(self):
        return self.log_alpha.exp()
    

    def act(self, obs, sample=False):
        dist = self.actor(obs) 
        action = dist.sample() if sample else dist.mean 
        # action = action.clamp(*self.action_range)
        
        return utils.to_np(action[0])
    

    def act_partner(self, obs, sample=False):
        dist = self.actor_partner(obs)
        action = dist.sample() if sample else dist.mean
        # action = action.clamp(*self.action_range)
        
        return utils.to_np(action[0])
    

    def update_critic(self, obs, action, reward, next_obs, done):  
        dist = self.actor(next_obs)
        next_action = dist.rsample() 
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + ((1 - done) * self.opt.gamma * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)  

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        
        q_value = torch.min(current_Q1, current_Q2).mean().detach().cpu().item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item(), q_value

    
    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
 

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.opt.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item()
            

    def update(self, replay_buffer, episode):
        obs, action, reward, next_obs, done = replay_buffer.sample(
            self.opt.batch_size)

        avg_reward = reward.mean().cpu().item() 
        critic_loss, q_value = self.update_critic(obs, action, reward, next_obs, done) 

        if episode % self.opt.actor_update_frequency == 0:
            actor_loss, alpha_loss = self.update_actor_and_alpha(obs)

        if episode % self.opt.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.opt.tau)
             
            
        self.avg_reward_list = np.append(self.avg_reward_list, avg_reward)
        self.actor_loss_list = np.append(self.actor_loss_list, actor_loss)
        self.critic_loss_list = np.append(self.critic_loss_list, critic_loss)
        self.alpha_loss_list = np.append(self.alpha_loss_list, alpha_loss)
        self.q_value_list = np.append(self.q_value_list, q_value)
            

    def save(self, episode, buffer, sync):
        self.graph(episode)

        torch.save(self.actor.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_{episode}.pth')
        torch.save(self.actor_partner.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_partner_{episode}.pth')    
        torch.save(self.critic.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_{episode}.pth')  
        torch.save(self.log_alpha, f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/log_alpha_{episode}.pth') 
        torch.save(self.actor_optimizer.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_optim_{episode}.pth') 
        torch.save(self.critic_optimizer.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_optim_{episode}.pth')
        torch.save(self.log_alpha_optimizer.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/log_alpha_optim_{episode}.pth')
             
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/sync {sync}', self.alpha_loss_list)

        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/avg_reward_list_{episode}', self.avg_reward_list)
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_loss_list_{episode}', self.actor_loss_list)
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_loss_list_{episode}', self.critic_loss_list)
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/alpha_loss_list_{episode}', self.alpha_loss_list)
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/q_value_list_{episode}', self.q_value_list)    
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/avg_return_list_{episode}', self.avg_return_list)
         

        with open(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/buffer_{episode}.pkl', 'wb') as f: # replay_buffer.pkl
            pickle.dump(buffer, f)


    def graph(self, episode):
        for lists, labels in [[self.avg_reward_list, f'Avg Reward'], 
                              [self.actor_loss_list, f'Actor Loss'], 
                              [self.critic_loss_list, f'Critic Loss'], 
                              [self.alpha_loss_list, f'Alpha Loss'],
                              [self.q_value_list, f'Q-Value'],
                              [self.avg_return_list, f'Avg Return']]:
            temp_x = list(range(0, len(lists)))

            plt.plot(temp_x, lists, label = f'{labels}')
            plt.legend()

            plt.savefig(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/graph/{labels}_{episode}')
            plt.close() 
