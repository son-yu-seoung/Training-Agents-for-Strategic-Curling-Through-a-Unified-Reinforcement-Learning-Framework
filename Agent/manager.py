from Agent.SAC.SAC import SACAgent, ReplayBuffer
from Agent.SAC.actor import *
from Agent.SAC.critic import *

from Agent.A2C.A2C import A2CAgent 
from Agent.A2C.actor import *
from Agent.A2C.critic import *

from Agent.PPO.PPO import PPOAgent 
from Agent.PPO.actor import *
from Agent.PPO.critic import *


def select_model(opt):
    if opt.agent_model == "SAC":
        return SACAgent(opt), ReplayBuffer(opt.obs_shape, opt.action_shape, opt.capacity)
    elif opt.agent_model == "A2C":
        return A2CAgent(opt), None
    elif opt.agent_model == "PPO":
        return PPOAgent(opt), None
