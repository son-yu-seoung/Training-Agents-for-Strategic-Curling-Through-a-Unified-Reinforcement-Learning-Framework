import numpy as np
from simulator import CurlingSimulator 
from GUI_utils.stone import Stone
import time
import math

class Environment(CurlingSimulator):
    def __init__(self, opt):
        self.opt = opt

        super().__init__(opt)
    

    def get_stone(self, action):
        self.delta_P.x, self.delta_P.y = self.calculate_delta(action[0], action[1]) # force, angle 
        force = self.delta_P.magnitude() * self.scale 
        self.delta_P.normalize() 
        initCondVec = self.delta_P * force   

        stone = Stone(self.start_P.x, self.start_P.y, r=self.radius, c = (255, 0, 0))
        stone.vel.x, stone.vel.y = initCondVec.x, initCondVec.y    
        stone.initial_angleVel = action[2] # angle velocity
        stone.scale = self.scale 

        return stone


    def check_out(self, prev_stone_list_red, prev_stone_list_yellow):
        def is_out(stone):
            x, y, r = stone.pos.x, stone.pos.y, stone.radius

            if x - r < self.hog_right or x - r > self.back_line_right: 
                return True
             
            if y - r < self.margin_top or y + r > self.margin_top + self.nY:
                return True
        
            return False
        
        def is_FGZ(idx, r, prev_stone_list):
            x, y = prev_stone_list[idx][0], prev_stone_list[idx][1]
            house_x, house_y = self.tee_line_right, self.center_line
            # prev로 부터 얻어오는 xyr이기 때문에 무조건 유효 영역 내 스톤이 있다는 가정
            if x + r > self.tee_line_right: 
                return False
            
            if math.sqrt((house_x - x) ** 2 + (house_y - y) ** 2) > r + self.house_right[3]:
                return True

        def distance_from_house(stone):
            house_x, house_y = self.tee_line_right, self.center_line
            return math.sqrt((house_x - stone.pos.x) ** 2 + (house_y - stone.pos.y) ** 2)
        
        reward = 0

        if self.opt.phase == 1:
            max_dist = self.tee_line_right - self.hog_left
            dist = distance_from_house(self.stone)
            reward = (max_dist - dist) / max_dist # 거리에 따른 -1 ~ 0 패널티 
        
        if not is_out(self.stone):
            reward += 0.15
             
        if self.stones_fired_red + self.stones_fired_yellow <= 4:
            # 파이브 가드 확인
            first = True
            if self.current_team == 0:
                for idx, stone in enumerate(self.sm.stone_list_yellow):
                    if is_out(stone) and is_FGZ(idx, stone.radius, prev_stone_list_yellow):
                        if first:
                            self.sm.stone_list_red.pop()
                            first = False
                            if self.opt.phase == 2:
                                reward = -1

                        stone.pos.x = prev_stone_list_yellow[idx][0]
                        stone.pos.y = prev_stone_list_yellow[idx][1]
            
            else:
                for idx, stone in enumerate(self.sm.stone_list_red):   
                    if is_out(stone) and is_FGZ(idx, stone.radius, prev_stone_list_red):
                        if first:
                            self.sm.stone_list_yellow.pop()
                            first = False
                            if self.opt.phase == 2:
                                reward = -1

                        stone.pos.x = prev_stone_list_red[idx][0]
                        stone.pos.y = prev_stone_list_red[idx][1]

        for idx, stone in enumerate(self.sm.stone_list_red):
            if is_out(stone):
                self.sm.stone_list_red.pop(idx)

        for idx, stone in enumerate(self.sm.stone_list_yellow):
            if is_out(stone):
                self.sm.stone_list_yellow.pop(idx) 
         
        return reward


    def shot(self, action):   
        prev_stone_list_red, prev_stone_list_yellow = [], []

        for stone in self.sm.stone_list_red: 
            prev_stone_list_red.append([stone.pos.x, stone.pos.y])
            
        for stone in self.sm.stone_list_yellow: 
            prev_stone_list_yellow.append([stone.pos.x, stone.pos.y])

        self.stone = self.get_stone(action)
        self.sm.add(self.stone, self.current_team) 
 
        self.is_stop = False

        while True:
            all_stop = True
            time.sleep(1)

            for stone in (self.sm.stone_list_red + self.sm.stone_list_yellow):
                if not(stone.vel.x == 0 and stone.vel.y == 0):
                    all_stop = False
            
            if all_stop:
                break

        reward = self.check_out(prev_stone_list_red, prev_stone_list_yellow) # -1~0.3

        prev_score_red    = self.score_red
        prev_score_yellow = self.score_yellow

        self.score_red, self.score_yellow = self.scoring()

        if self.opt.phase == 2:
            if self.current_team == 0: # red
                prev_diff = prev_score_red - prev_score_yellow
                curr_diff = self.score_red - self.score_yellow

                is_last_shot = (self.stones_fired_red == 8)

            else: # yellow
                prev_diff = prev_score_yellow - prev_score_red
                curr_diff = self.score_yellow - self.score_red

                is_last_shot = (self.stones_fired_yellow == 8)

            delta_diff = curr_diff - prev_diff # 이번 샷으로 점수차가 얼마나 변했는지
            weight = 1.0 if is_last_shot else 0.5
            reward += weight * delta_diff 

        done = 0
        if self.stones_fired_red == 8 or self.stones_fired_yellow == 8:
            done = 1
 
        # reward = self.scale_reward(reward)

        return reward, done
    

    def scale_reward(self, reward, min_r=-2, max_r=2):
        """보상을 -1 ~ 1 범위로 스케일링"""
        scaled_reward = 2 * ((reward - min_r) / (max_r - min_r)) - 1
        scaled_reward * 10 

        return np.clip(scaled_reward, -10, 10)


    def btn_auto(self):
        pass


    def btn_random(self):
        pass


    def btn_shot(self):
        pass