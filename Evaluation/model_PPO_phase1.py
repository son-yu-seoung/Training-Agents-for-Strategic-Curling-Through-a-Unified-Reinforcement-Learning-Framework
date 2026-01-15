from Agent.PPO.PPO import PPOAgent
from Agent.PPO.actor import *
from Agent.PPO.critic import *
import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="PPO_phase1_v1.1_V(s)")
    parser.add_argument("--s_episode", type=int, default=2000) # episode_A 
    parser.add_argument("--agent_model", type=str, default="PPO")   
    
    parser.add_argument("--phase", type=int, default=1)

    parser.add_argument("--obs_shape", type=tuple, default=(16, 8)) # coord + turn
    parser.add_argument("--obs_dim", type=int, default=128)

    parser.add_argument("--action_shape", type=tuple, default=(3,)) 
    parser.add_argument("--action_dim", type=int, default=3) 
    parser.add_argument("--hidden_dim", type=int, default=256) 
    parser.add_argument("--hidden_depth", type=int, default=2) 
    parser.add_argument("--log_std_bounds", type=list, default=[-2, 1]) 

    # parser.add_argument("--learnable_temperature", type=bool, default=True) 
    # parser.add_argument("--init_temperature", type=float, default=0.2) 
    parser.add_argument("--lr", type=float, default=3e-4) 
    parser.add_argument("--beta", type=list, default=[0.9, 0.999]) 
    parser.add_argument("--tau", type=float, default=0.005) # 작을수록 업데이트가 천천히 (안전성)
    parser.add_argument("--gamma", type=float, default=0.9) 

    parser.add_argument("--ppo_clip", type=float, default=0.2) 
    parser.add_argument("--ppo_epochs", type=float, default=10) 
    parser.add_argument("--ppo_ent_coef", type=float, default=0.0) 

    new_opt = parser.parse_args()

    return new_opt

class Model_PPO_Phase1:
    def __init__(self, env, opt):
        self.env = env
        self.opt = opt
        new_opt = config()

        self.opt.__dict__.update(new_opt.__dict__) 
        
        self.agent = PPOAgent(self.opt)
        self.agent.network_init()

        self.agent.actor.eval()
        self.agent.actor_partner.eval()
        self.agent.critic.eval()


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
    

    def get_action(self, obs): 
        obs = torch.tensor(obs, dtype=torch.float32).cuda()    

        action, log_prob, value = self.agent.act(obs, sample=True) # deterministic 
        action_np = action.detach().cpu().numpy().reshape(-1)

        scale_action = [(action_np[0] + 1) * 4, action_np[1] * self.env.angle_range, action_np[2] * 1.5] 

        return action, scale_action


    def play(self, turn):
        obs = self.get_state(turn)
        action, scale_action = self.get_action(np.copy(obs))

        return scale_action