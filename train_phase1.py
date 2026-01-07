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

    parser.add_argument("--phase", type=int, default=1)  
    parser.add_argument("--ratio", type=float, default=0.4)
    parser.add_argument("--game_speed", type=int, default=2000)

    parser.add_argument("--s_episode", type=int, default=0)
    parser.add_argument("--e_episode", type=int, default=2000)

    parser.add_argument("--version", type=str, default='phase1_v4.1_returns')  
    parser.add_argument("--agent_model", type=str, default='SAC')  
    parser.add_argument("--capacity", type=int, default=4000)
    parser.add_argument("--start_update", type=int, default=1000) # 5000 shot 기준 

    parser.add_argument("--obs_shape", type=tuple, default=(16, 8)) # pos, ctx, game
    parser.add_argument("--obs_dim", type=int, default=128)

    parser.add_argument("--action_shape", type=tuple, default=(3,))  
    parser.add_argument("--action_dim", type=int, default=3) 
    parser.add_argument("--hidden_dim", type=int, default=256) 
    parser.add_argument("--hidden_depth", type=int, default=2) 
    parser.add_argument("--log_std_bounds", type=list, default=[-5, 2]) 

    parser.add_argument("--batch_size", type=int, default=256) # 512 

    parser.add_argument("--lr", type=float, default=3e-4) 
    parser.add_argument("--beta", type=list, default=[0.9, 0.999]) 
    parser.add_argument("--tau", type=float, default=0.005) # 작을수록 업데이트가 천천히 (안전성)
    parser.add_argument("--gamma", type=float, default=0.94) 
    parser.add_argument("--learnable_temperature", type=bool, default=True) 
    parser.add_argument("--init_temperature", type=float, default=0.2) 

    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8) 
    
    parser.add_argument("--actor_update_frequency", type=float, default=1) 
    parser.add_argument("--critic_target_update_frequency", type=float, default=10) 
    parser.add_argument("--partner_update", type=int, default=50) 

    cfg = parser.parse_args()

    return cfg


class Train():
    def __init__(self):
        super().__init__()
        seed_everything(cfg.seed)
        self.makedirs()

        self.env = Environment(cfg)

        self.agent, self.buffer = select_model(cfg)
        self.agent.network_init()
        self.buffer_init()


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
        score_info = np.full((16, 1), 0, dtype=np.float32)

        obs = np.concatenate((coord_info, turn_info, score_info), axis=1)  
        obs = np.expand_dims(obs, axis=0) # (1, 16, 8) 

        return obs 
    

    def get_action(self, obs, partner=False):
        if len(self.buffer) < cfg.start_update:
            rand_force   = random.uniform(-1, 1)
            rand_angle   = random.uniform(-1, 1)  
            rand_ang_vel = random.uniform(-1, 1)

            action       = [rand_force, rand_angle, rand_ang_vel]
            scale_action = [(action[0] + 1) * 4, action[1] * self.env.angle_range, action[2] * 1.5] 

            return action, scale_action
        
        else:
            obs = torch.tensor(obs, dtype=torch.float32).cuda()

            if partner:
                action = self.agent.act_partner(obs, sample=True)
            else:
                action = self.agent.act(obs, sample=True)  
            scale_action = [(action[0] + 1) * 4, action[1] * self.env.angle_range, action[2] * 1.5] 

            return action, scale_action


    def roll_out(self, episode):
        self.env.reset()
        self.env.current_team = random.randint(0, 1) 
        self.episode_reward = 0.0

        obs = self.get_state(turn=0)

        for n_shot in range(16): 
            if self.env.current_team == 0:
                self.env.stones_fired_red += 1

                self.action, self.scale_action = self.get_action(np.copy(obs))
                reward, done = self.env.shot(self.scale_action)
                
                self.episode_reward += reward

                self.env.current_team = 1 - self.env.current_team
                next_obs = self.get_state(turn=n_shot+1)

                self.buffer.add(obs, self.action, reward, next_obs, done)
                obs = next_obs
                
            else:
                self.env.stones_fired_yellow += 1

                action, scale_action = self.get_action(np.copy(obs), partner=True)
                reward, _ = self.env.shot(scale_action)

                self.env.current_team = 1 - self.env.current_team
                next_obs = self.get_state(turn=n_shot+1)

                obs = next_obs
             
            if len(self.buffer) >= cfg.start_update:
                self.agent.update(self.buffer, episode)

            
            print(f'Episode {episode} (sync {self.sync}), Turn [{n_shot+1}/{16}], Buffer [{len(self.buffer)}/{cfg.capacity}], Score(R-Y) {self.env.score_red}-{self.env.score_yellow}, Reward {reward:.4f} (Team {1 - self.env.current_team})', end=' ||| ')
            try:
                print(f'Returns {self.agent.return_list[-1]:.4f}, Q-value {self.agent.q_value_list[-1]:.4f}, Avg Reward {self.agent.avg_reward_list[-1]:.4f}')
            except:
                print(f'Returns None, Q-value None, Avg Reward None')

        self.agent.log_episode_return(self.episode_reward)
        
        return self.env.score_red, self.env.score_yellow


    def train(self):
        time.sleep(1)
        self.sync = 0

        for episode in range(cfg.s_episode, cfg.e_episode + 1):
            red_score, yellow_score = self.roll_out(episode)

            # if self.win_rate >= opt.partner_update_rate and len(self.agent.avg_reward_list) != 0: 
            if episode != 0 and episode % cfg.partner_update == 0:
                self.agent.actor_partner.load_state_dict(self.agent.actor.state_dict())
                self.sync += 1

            if episode != 0 and episode % cfg.save_interval == 0:
                self.agent.save(episode, self.buffer, self.sync)
        
        return 0

    
    def buffer_init(self):
        if cfg.phase == 1:
            if cfg.s_episode != 0:
                with open(f'./Agent/{cfg.agent_model}/save/{cfg.version}/weights/buffer_{cfg.s_episode}.pkl', 'rb') as f:
                    self.buffer = pickle.load(f)


if __name__ == '__main__':
    cfg = config()

    trainer = Train()
    
    thread = threading.Thread(target=trainer.train, daemon=True)
    thread.start()


    trainer.env.run()
