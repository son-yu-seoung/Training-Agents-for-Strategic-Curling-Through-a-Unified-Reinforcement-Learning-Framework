import os
os.environ["PYTHONHASHSEED"] = str(42)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  

import torch
import argparse
import random
import pickle
import time

from snippet.seed import *

import threading
from environment import Environment

from Agent.manager import select_model


def config():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # gpu id 
    print(torch.cuda.is_available())
    print()

    parser = argparse.ArgumentParser()

    parser.add_argument("--phase", type=int, default=2)  
    parser.add_argument("--ratio", type=float, default=0.4)
    parser.add_argument("--game_speed", type=int, default=2000)

    parser.add_argument("--s_episode", type=int, default=0)
    parser.add_argument("--e_episode", type=int, default=2000)

    parser.add_argument("--version", type=str, default='PPO_phase2_v1.0')  
    parser.add_argument("--agent_model", type=str, default='PPO')   
    # parser.add_argument("--start_update_epi", type=int, default=125) # 5000 shot 기준 

    parser.add_argument("--phase1_version", type=str, default='PPO_phase1_v1.1_V(s)')
    parser.add_argument("--phase1_episode", type=int, default=2000)

    parser.add_argument("--obs_shape", type=tuple, default=(16, 8)) # pos, ctx, game
    parser.add_argument("--obs_dim", type=int, default=128)

    parser.add_argument("--action_shape", type=tuple, default=(3,))  
    parser.add_argument("--action_dim", type=int, default=3) 
    parser.add_argument("--hidden_dim", type=int, default=256) 
    parser.add_argument("--hidden_depth", type=int, default=2) 
    parser.add_argument("--log_std_bounds", type=list, default=[-2, 1]) 

    parser.add_argument("--batch_size", type=int, default=32) # 512 

    parser.add_argument("--update_epi", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--beta", type=list, default=[0.9, 0.999]) 
    parser.add_argument("--tau", type=float, default=0.005) # 작을수록 업데이트가 천천히 (안전성)
    parser.add_argument("--gamma", type=float, default=0.97) 
    # parser.add_argument("--learnable_temperature", type=bool, default=True) 
    # parser.add_argument("--init_temperature", type=float, default=0.2) 

    parser.add_argument("--ppo_clip", type=float, default=0.2) 
    parser.add_argument("--ppo_epochs", type=float, default=10) 
    parser.add_argument("--ppo_ent_coef", type=float, default=0.0) 

    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8) 
    
    # parser.add_argument("--actor_update_frequency", type=float, default=1) 
    # parser.add_argument("--critic_target_update_frequency", type=float, default=10) 
    parser.add_argument("--partner_update", type=int, default=50) 

    cfg = parser.parse_args()

    return cfg


class Train():
    def __init__(self):
        super().__init__()
        seed_everything(cfg.seed)
        self.makedirs()

        self.env = Environment(cfg)

        self.agent, _ = select_model(cfg)
        self.agent.network_init() 

        self.traj_buffer = []


    def makedirs(self):
        os.makedirs(f'./Agent/{cfg.agent_model}/save/{cfg.version}/weights', exist_ok=True)
        os.makedirs(f'./Agent/{cfg.agent_model}/save/{cfg.version}/graph', exist_ok=True)


    def get_state(self, turn): 
        coord_info_red = np.full((8, 6), -1, dtype=np.float32) # (x, y, team, v_x, v_y, ||v||)
        coord_info_yellow = np.full((8, 6), -1, dtype=np.float32) # (x, y, team, v_x, v_y, ||v||)
        norm_x, norm_y = self.env.nX, self.env.nY 
 
        for idx, stone in enumerate(self.env.sm.stone_list_red):
            x, y   = stone.pos.x / norm_x           , (stone.pos.y - self.env.margin_top) / norm_y
            x0, y0 = self.env.start_P.x / norm_x, (self.env.start_P.y - self.env.margin_top) / norm_y

            dx = x - x0
            dy = y - y0

            dist = np.sqrt(dx**2 + dy**2).astype(np.float32) 
            if dist < 1e-6:
                dx, dy = 0.0, 0.0
                dist = 0.0
            else:
                dx, dy = dx / dist, dy / dist

            if self.env.current_team == 0:
                coord_info_red[idx] = [x, y, 0, dx, dy, dist]
            else:
                coord_info_red[idx] = [x, y, 1, dx, dy, dist]
 
        for idx, stone in enumerate(self.env.sm.stone_list_yellow):
            x, y   = stone.pos.x / norm_x           , (stone.pos.y - self.env.margin_top) / norm_y
            x0, y0 = self.env.start_P.x / norm_x, (self.env.start_P.y - self.env.margin_top) / norm_y

            dx = x - x0
            dy = y - y0

            dist = np.sqrt(dx**2 + dy**2).astype(np.float32) 
            if dist < 1e-6:
                dx, dy = 0.0, 0.0
                dist = 0.0
            else:
                dx, dy = dx / dist, dy / dist

            if self.env.current_team == 0:
                coord_info_yellow[idx] = [x, y, 1, dx, dy, dist]
            else:
                coord_info_yellow[idx] = [x, y, 0, dx, dy, dist]

        if self.env.current_team == 0: # red
            coord_info = np.concatenate((coord_info_red, coord_info_yellow), axis=0)

        else: # yellow
            coord_info = np.concatenate((coord_info_yellow, coord_info_red), axis=0)

        turn_info = np.zeros(16, dtype=np.float32)
        
        if turn != 16:
            turn_info[turn] = 1 
        
        turn_info = np.expand_dims(turn_info, axis=-1)

        if self.env.current_team == 0:
            lead = self.env.score_red - self.env.score_yellow
        else:
            lead = self.env.score_yellow - self.env.score_red
        
        score_info = np.full((16, 1), lead, dtype=np.float32)

        obs = np.concatenate((coord_info, turn_info, score_info), axis=1)  
        obs = np.expand_dims(obs, axis=0) # (1, 16, 8) 

        return obs 
    

    def get_action(self, obs, partner=False): 
        obs = torch.tensor(obs, dtype=torch.float32).cuda()

        if partner:
            action = self.agent.act_partner(obs, sample=True)
            action_np = action.detach().cpu().numpy().reshape(-1)

            scale_action = [(action_np[0] + 1) * 4, action_np[1] * self.env.angle_range, action_np[2] * 1.5] 

            return action, scale_action
        
        else:
            action, log_prob, value = self.agent.act(obs, sample=True)   
            action_np = action.detach().cpu().numpy().reshape(-1)

            scale_action = [(action_np[0] + 1) * 4, action_np[1] * self.env.angle_range, action_np[2] * 1.5] 

            return action, scale_action, log_prob, value


    def roll_out(self, episode):
        self.env.reset()
        self.env.current_team = random.randint(0, 1) 
        self.episode_reward = 0.0

        obs = self.get_state(turn=0)

        traj = { 
            'obs'      : [],
            'actions'  : [],
            'log_probs': [],
            'values'   : [],
            'rewards'  : [],
            'dones'    : [],
        }

        for n_shot in range(16): 
            if self.env.current_team == 0:
                self.env.stones_fired_red += 1 

                action, scale_action, log_prob, value = self.get_action(np.copy(obs)) 
                reward, done = self.env.shot(scale_action)

                self.episode_reward += reward

                self.env.current_team = 1 - self.env.current_team
                next_obs = self.get_state(turn=n_shot+1)

                # trajectory 저장 (우리 샷만)  
                traj['obs'].append(obs[0])
                traj['actions'].append(action.detach().cpu().numpy())
                traj['log_probs'].append(log_prob.detach())
                traj['values'].append(value.detach())
                traj['rewards'].append(reward)
                traj['dones'].append(done)

                obs = next_obs
                
            else:
                self.env.stones_fired_yellow += 1

                action, scale_action = self.get_action(np.copy(obs), partner=True)
                reward, _ = self.env.shot(scale_action)

                self.env.current_team = 1 - self.env.current_team
                next_obs = self.get_state(turn=n_shot+1)

                obs = next_obs
            
            print(f'Episode {episode} (sync {self.sync}), Turn [{n_shot+1}/{16}], Score(R-Y) {self.env.score_red}-{self.env.score_yellow}, Reward {reward:.4f} (Team {1 - self.env.current_team})', end=' ||| ')
            try:
                print(f'State Value {self.agent.state_value_list[-1]:.4f}, Returns {self.agent.avg_return_list[-1]:.4f}, Avg Reward {self.agent.avg_reward_list[-1]:.4f}')
            except:
                print(f'State Value None, Returns None, Avg Reward None')

        
        # 에피소드 끝나고 한 번 A2C 업데이트 
        self.agent.log_episode_return(self.episode_reward)
        # self.agent.update(traj)

        
        return self.env.score_red, self.env.score_yellow, traj


    def train(self):
        time.sleep(1)
        self.sync = 0

        for episode in range(cfg.s_episode, cfg.e_episode + 1):
            red_score, yellow_score, traj = self.roll_out(episode)
            self.traj_buffer.append(traj)

            if len(self.traj_buffer) >= cfg.update_epi:
                merged_traj = {
                    'obs'      : [],
                    'actions'  : [],
                    'log_probs': [],
                    'values'   : [],
                    'rewards'  : [],
                    'dones'    : [],
                }
                for epi_traj in self.traj_buffer:
                    merged_traj["obs"].extend(epi_traj["obs"])
                    merged_traj["actions"].extend(epi_traj["actions"])
                    merged_traj["log_probs"].extend(epi_traj["log_probs"])
                    merged_traj["values"].extend(epi_traj["values"])
                    merged_traj["rewards"].extend(epi_traj["rewards"])
                    merged_traj["dones"].extend(epi_traj["dones"])

                self.agent.update(merged_traj)
                self.traj_buffer.clear()

            # if self.win_rate >= opt.partner_update_rate and len(self.agent.avg_reward_list) != 0: 
            if episode != 0 and episode % cfg.partner_update == 0:
                self.agent.actor_partner.load_state_dict(self.agent.actor.state_dict())
                self.sync += 1

            if episode != 0 and episode % cfg.save_interval == 0: 
                self.agent.save(episode, self.sync)
        
        return 0

    
    def buffer_init(self):
        pass


if __name__ == '__main__':
    cfg = config()

    trainer = Train()
    
    thread = threading.Thread(target=trainer.train, daemon=True)
    thread.start()

    trainer.env.run()