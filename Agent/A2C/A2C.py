import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Agent.A2C.actor import *
from Agent.A2C.critic import *
 
import utils
import pickle

# SAC 주요 기술 적용:
# - CNN을 활용한 상태 인코딩 (컬링 환경에서 state가 이미지이므로 CNN 사용)
# - Twin Q-Network 사용 (Q-value 과대평가 방지)
# - Stochastic Policy (확률적 정책 사용)
# - Entropy Regularization (탐색 강화)

class A2CAgent():
    def __init__(self, opt): 
        self.opt = opt 
 
        self.actor = DiagGaussianActor(opt).cuda()
        self.actor_partner = DiagGaussianActor(opt).cuda()
        self.actor_partner.load_state_dict(self.actor.state_dict())

        self.critic = ValueCritic(opt).cuda()  

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=opt.lr,
                                                betas=opt.beta)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
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

                self.actor_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_optim_{self.opt.s_episode}.pth'))
                self.critic_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_optim_{self.opt.s_episode}.pth')) 


                self.avg_reward_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/avg_reward_list_{self.opt.s_episode}.npy')
                self.actor_loss_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_loss_list_{self.opt.s_episode}.npy')
                self.critic_loss_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_loss_list_{self.opt.s_episode}.npy') 
                self.state_value_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/state_value_list_{self.opt.s_episode}.npy')
                self.avg_return_list = np.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/avg_return_list_{self.opt.s_episode}.npy')

            else:
                self.avg_reward_list = np.array([]) 
                self.actor_loss_list = np.array([]) 
                self.critic_loss_list = np.array([]) 
                self.state_value_list = np.array([]) 
                self.avg_return_list = np.array([])
        
        if self.opt.phase == 2:
            if eval:
                self.actor.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_{self.opt.s_episode}.pth'))
                self.actor_partner.load_state_dict(self.actor.state_dict())  
                self.critic.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_{self.opt.s_episode}.pth')) 
                   
            else:
                self.actor.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/actor_{self.opt.phase1_episode}.pth'))
                self.actor_partner.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/actor_partner_{self.opt.phase1_episode}.pth')) 

            # self.critic.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/critic_{self.opt.phase1_episode}.pth'))  

            # self.log_alpha = torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/log_alpha_{self.opt.phase1_episode}.pth')
            # self.log_alpha.requires_grad = True 

            # self.actor_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/actor_optim_{self.opt.phase1_episode}.pth'))
            # self.critic_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/critic_optim_{self.opt.phase1_episode}.pth'))
            # self.log_alpha_optimizer.load_state_dict(torch.load(f'./Agent/{self.opt.agent_model}/save/{self.opt.phase1_version}/weights/log_alpha_optim_{self.opt.phase1_episode}.pth'))
    
            self.avg_reward_list = np.array([]) 
            self.actor_loss_list = np.array([]) 
            self.critic_loss_list = np.array([]) 
            self.state_value_list = np.array([])
            self.avg_return_list = np.array([])
    
    def log_episode_return(self, episode_return):
        self.avg_return_list = np.append(self.avg_return_list, episode_return)


    def act(self, obs, sample=False): 
        dist = self.actor(obs) 
        action = dist.sample() if sample else dist.mean 
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) # (B, 1)?
        value = self.critic(obs)
        
        return action[0], log_prob, value
    

    def act_partner(self, obs, sample=False):
        dist = self.actor_partner(obs)
        action = dist.sample() if sample else dist.mean

        return action[0]
    

    def update(self, trajectory):  
        log_probs = torch.cat([lp.view(1, 1) for lp in trajectory['log_probs']], dim=0)           # (T, 1)
        values    = torch.cat([v.view(1, 1)  for v  in trajectory['values']     ], dim=0)         # (T, 1)
        rewards   = torch.tensor(trajectory['rewards'], dtype=torch.float32).cuda().unsqueeze(-1) # (T, 1)
        dones     = torch.tensor(trajectory['dones'], dtype=torch.float32).cuda().unsqueeze(-1)   # (T, 1)

        T = len(dones)
 
        # log_probs = torch.cat(log_probs_list, dim=0)  # (T, 1)
        # values = torch.cat(values_list, dim=0)        # (T, 1)
        # rewards = torch.cat(rewards_list, dim=0)      # (T, 1)
        # dones = torch.cat(dones_list, dim=0)          # (T, 1)

        returns = []
        R = torch.zeros(1, 1).cuda()
        for t in reversed(range(T)):
            R = rewards[t] + self.opt.gamma * R * (1.0 - dones[t])
            returns.insert(0, R)
        returns = torch.cat(returns, dim=0).detach() # (T, 1)

        # Advantage
        advantages = returns - values.detach()

        # Actor Loss
        actor_loss = -(log_probs * advantages).mean()

        # Critic Loss
        critic_loss = F.mse_loss(values, returns)

        # optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() 

        actor_loss_scalar = actor_loss.detach().cpu().item()
        critic_loss_scalar = critic_loss.detach().cpu().item()
        avg_reward = rewards.mean().detach().cpu().item()  

        avg_state_value = values.mean().detach().cpu().item()
            
        self.avg_reward_list = np.append(self.avg_reward_list, avg_reward)
        self.actor_loss_list = np.append(self.actor_loss_list, actor_loss_scalar)
        self.critic_loss_list = np.append(self.critic_loss_list, critic_loss_scalar) 
        self.state_value_list = np.append(self.state_value_list, avg_state_value)  
            

    def save(self, episode, sync): 
        self.graph(episode) 

        torch.save(self.actor.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_{episode}.pth')
        torch.save(self.actor_partner.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_partner_{episode}.pth')    
        torch.save(self.critic.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_{episode}.pth')   

        torch.save(self.actor_optimizer.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_optim_{episode}.pth') 
        torch.save(self.critic_optimizer.state_dict(), f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_optim_{episode}.pth') 
              
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/avg_reward_list_{episode}', self.avg_reward_list)
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/actor_loss_list_{episode}', self.actor_loss_list)
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/critic_loss_list_{episode}', self.critic_loss_list)
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/state_value_list_{episode}', self.state_value_list)
        np.save(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/weights/avg_return_list_{episode}', self.avg_return_list)
        

    def graph(self, episode):
        for lists, labels in [[self.avg_reward_list, f'Avg Reward'], 
                              [self.actor_loss_list, f'Actor Loss'], 
                              [self.critic_loss_list, f'Critic Loss'],
                              [self.state_value_list, f'State Value'],  
                              [self.avg_return_list, f'Avg Return']]:
            temp_x = list(range(0, len(lists)))

            plt.plot(temp_x, lists, label = f'{labels}')
            plt.legend()

            plt.savefig(f'./Agent/{self.opt.agent_model}/save/{self.opt.version}/graph/{labels}_{episode}')
            plt.close() 

