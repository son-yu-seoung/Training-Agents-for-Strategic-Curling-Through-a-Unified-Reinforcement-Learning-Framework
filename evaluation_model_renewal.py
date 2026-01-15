import os
os.environ["PYTHONHASHSEED"] = str(42)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  

import torch
import threading 
import argparse 
import pandas as pd
import numpy as np
import random
import time
import copy

from snippet.seed import *

from environment import Environment
from Agent.manager import select_model

from Evaluation.model_phase1 import Model_Phase1 
from Evaluation.model_phase2 import Model_Phase2  
from Evaluation.model_phase_all import Model_Phase_All  
from Evaluation.model_A2C_phase1 import Model_A2C_Phase1 
from Evaluation.model_A2C_phase2 import Model_A2C_Phase2
from Evaluation.model_PPO_phase1 import Model_PPO_Phase1 
from Evaluation.model_PPO_phase2 import Model_PPO_Phase2


def config():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # gpu id 
    print(torch.cuda.is_available()) 

    parser = argparse.ArgumentParser()  
    parser.add_argument("--n_game", type=int, default=30) 
    parser.add_argument("--ends", type=int, default=10) # test
    parser.add_argument("--turns", type=int, default=16)  
    
    parser.add_argument("--ratio", type=float, default=0.6)  
    parser.add_argument("--game_speed", type=int, default=2000)  
    
    parser.add_argument("--seed", type=int, default=8) 

  
    cfg = parser.parse_args()

    return cfg


class Evaluation:
    def __init__(self): 
        seed_everything(cfg.seed)

        cfg_B = copy.deepcopy(cfg)
        self.env = Environment(cfg) # btn 자동 비활성화 1

        self.modelA = Model_A2C_Phase1(self.env, cfg) # 주체
        self.modelB = Model_PPO_Phase1(self.env, cfg_B) # 비교 대상

        self.makedirs()

        self.column_names = ['modelA (episode)', 'modelB (episode)', 'n_game', 'ends', 'win (A-B)', 'avg score (A)', 'avg score (B)']
        self.df = pd.DataFrame(columns=self.column_names)  
    

    def makedirs(self):
        os.makedirs('./Evaluation/csv', exist_ok=True)


    def cal_win(self, ends):
        scores_A = 0
        scores_B = 0

        for team, score in ends: 
            if team == 'A':
                scores_A += score
            elif team == 'B':
                scores_B += score 

        if scores_A > scores_B:
            win = 'A'
        elif scores_A < scores_B:
            win = 'B'
        else:
            win = 'C'

        return win, scores_A, scores_B


    def evaluate(self):
        time.sleep(1)

        for n_game in range(cfg.n_game):
            seed_everything(n_game)
            self.env.current_team = n_game % 2 # 번갈아가며 선공/후공

            avg_score_A = 0 
            avg_score_B = 0 

            end = 0
            ends = []
            
            while True: # 10엔드가 무승부로 끝나는 경우는 없기 때문에 처리 필요
                print('========================\n\n')
                print(f'{n_game+1}게임 {end+1}엔드 시작') 

                self.env.reset(eval=True)  
                # self.env.current_team = 1
                for t in range(cfg.turns): 
                    if self.env.current_team == 0: # model A
                        self.env.stones_fired_red += 1 
 
                        scale_action = self.modelA.play(turn=t)
                        _, _ = self.env.shot(scale_action)

                        self.env.current_team =  1 - self.env.current_team

                    else:
                        self.env.stones_fired_yellow += 1 
                        
                        action = self.modelB.play(turn=t)
                        _, _ = self.env.shot(action)

                        self.env.current_team =  1 - self.env.current_team


                if self.env.score_red > self.env.score_yellow:
                    ends.append(['A', self.env.score_red])
                
                elif self.env.score_red < self.env.score_yellow:
                    ends.append(['B', self.env.score_yellow])

                else :
                    ends.append(['C', 0])

                avg_score_A += self.env.score_red # 각 엔드의 score가 더해짐 
                avg_score_B += self.env.score_yellow # 각 엔드의 score가 더해짐 

                
                if self.env.score_red > self.env.score_yellow: # 후공이 이겼으니 순서변경
                    self.env.current_team = 0 # 잘한팀이 선공
                elif self.env.score_red < self.env.score_yellow:
                    self.env.current_team = 1
                else: # 무승부인 경우 선/후공 교체
                    self.env.current_team = 1 - self.env.current_team

                
                print(f'Red Team Score (modelA): {self.env.score_red}, Yellow Team Score (modelB): {self.env.score_yellow}')
                print('\n\n========================\n')

                end += 1
                if end >= 10:
                    win, _, _ = self.cal_win(ends)

                    if win != 'C':
                        break
            
            avg_score_A /= cfg.ends # 각 게임의 평균 score 
            avg_score_B /= cfg.ends # 각 게임의 평균 score 

            win, scores_A, scores_B = self.cal_win(ends)

            if len(self.df) == 0:
                df_row = {
                column_name: value
                for column_name, value in zip(self.column_names, [f'{self.modelA.opt.version} ({self.modelA.opt.s_episode})', f'{self.modelB.opt.version} ({self.modelB.opt.s_episode})', n_game+1, ends, f'{win} ({scores_A}-{scores_B})', avg_score_A, avg_score_B])
                }

            else:
                df_row = {
                column_name: value
                for column_name, value in zip(self.column_names[2:], [n_game+1, ends, f'{win} ({scores_A}-{scores_B})', avg_score_A, avg_score_B])
                }


            self.df.loc[len(self.df)] = df_row 
 
        self.df.to_csv(f'./Evaluation/csv/{self.modelA.opt.version} vs {self.modelB.opt.version}.csv')
        print('========\n')
        print('CSV 저장')
        print(f'./Evaluation/csv/{self.modelA.opt.version} vs {self.modelB.opt.version}.csv')
        print('\n========')


if __name__ == "__main__":
    cfg = config()

    eval = Evaluation() 
 
    thread = threading.Thread(target=eval.evaluate, daemon=True)
    thread.start()
    
    eval.env.run() 