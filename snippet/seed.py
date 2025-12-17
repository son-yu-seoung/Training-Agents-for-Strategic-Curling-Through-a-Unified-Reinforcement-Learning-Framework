import torch
import random
import os
import numpy as np


def seed_everything(seed): 
    random.seed(seed)
    np.random.seed(seed)
 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 
    torch.use_deterministic_algorithms(True) 
         

def seed_worker(worker_id): 
    worker_seed = torch.initial_seed() % 2 ** 32  
    worker_seed += worker_id
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)  

    torch.manual_seed(worker_seed)