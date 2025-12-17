import pygame
import pygame_widgets
from pygame_widgets.slider import Slider 
from pygame_widgets.button import Button
import threading

import time
import math 
import numpy as np

from GUI_utils.buttons import Buttons
from GUI_utils.utils import Utils
from GUI_utils.stone import Stone
from GUI_utils.point import Point
from GUI_utils.stone_manager import StoneManager

'''
Curling 경기장의 규격은 대한 컬링 연맹 "THE RULES OF CURLING and Rules of Competition"을 따른다.
'''

class CurlingSimulator(Buttons, Utils):
    def __init__(self, opt = None):
        self.opt = opt # from agent 
        self.init_scale(self.opt)   

        pygame.init() 
        pygame.font.init()
        self.font = pygame.font.SysFont("arial", int(40 * self.ratio), True, True)
         
        self.screen = pygame.display.set_mode((self.nX, self.nY + self.margin_top + self.margin_bottom))
        self.screen.fill((255, 255, 255))



        self.init_variables(eval=False)
        self.init_widgets()


    def init_scale(self, opt): 
        if opt == None:
            self.ratio = 0.4
        else:
            self.ratio = opt.ratio

        self.nX = 3076 * self.ratio
        self.nY = 475 * self.ratio 

        self.margin_top = 140 * self.ratio
        self.margin_bottom = 300 * self.ratio

        # vertical
        self.hog_left = 30 * self.ratio
        self.hog_right = 2224.8 * self.ratio
        self.tee_line_right = 2864.3 * self.ratio
        self.back_line_right = 3046.8 * self.ratio

        # horizontal
        self.center_line = self.margin_top + (self.nY // 2)
        self.wheelchair_line_right = 45.7 * self.ratio 

        # house
        self.house_right = [15.2 * self.ratio, 61 * self.ratio, 121.9 * self.ratio, 182.9 * self.ratio] 

        # score circle (x, y, r, red_half, yellow_half)
        self.score_circle = [(int((60 + i * 100) * self.ratio), self.margin_top//2, int(43 * self.ratio), False, False) for i in range(8)]


    def init_variables(self, eval):
        if not eval: # eval일 때는 current_team 초기화 X
            self.current_team = 0

        self.stones_fired_red = 0
        self.stones_fired_yellow = 0
        self.score_red = 0
        self.score_yellow = 0

        self.angle_range = self.calculate_angle((self.hog_left, self.center_line), (self.hog_right, self.margin_top))

        self.is_reset = False
        self.is_stop = True 
        self.is_auto = False
        self.state = False

        self.sm = StoneManager(self.screen, self.opt) 

        self.start_P = Point(self.hog_left, self.center_line)
        self.delta_P = Point() 

        self.radius = 13.9 * self.ratio
        self.stones = []
        self.scale = (self.hog_right - self.hog_left) / 50.0 


    def init_widgets(self):
        handle_radius = round(20 * self.ratio)
        self.slider_force = Slider(self.screen, int(150 * self.ratio), int(840 * self.ratio), int(250 * self.ratio), int(40 * self.ratio), min=0, max=8, step=0.001, colour=(255, 255, 255), handleRadius=handle_radius) 
        self.slider_angle = Slider(self.screen, int(20 * self.ratio), int(680 * self.ratio), int(40 * self.ratio), int(180 * self.ratio), min=0, max=2 * self.angle_range, colour=(255, 255, 255), handleRadius=handle_radius, step=0.01, vertical=True)   
        self.slider_ang_vel = Slider(self.screen, int(80 * self.ratio), int(680 * self.ratio), int(40 * self.ratio), int(180 * self.ratio), min=0, max=3 , colour=(255, 255, 255), handleRadius=handle_radius, step=0.01, vertical=True)   

        self.button_auto = Button(self.screen, int(440 * self.ratio), int(630 * self.ratio), int(140 * self.ratio), int(80 * self.ratio), text='auto', fontSize=int(50 * self.ratio), onClick=self.btn_auto)
        self.button_random = Button(self.screen, int(440 * self.ratio), int(725 * self.ratio), int(140 * self.ratio), int(80 * self.ratio), text='random', fontSize=int(50 * self.ratio), onClick=self.btn_random)
        self.button_shot = Button(self.screen, int(440 * self.ratio), int(820 * self.ratio), int(140 * self.ratio), int(80 * self.ratio), text='shot', fontSize=int(50 * self.ratio), onClick=self.btn_shot)
 

    def reset(self, eval=False):   
        self.init_variables(eval) 

        # (x, y, r, red_half, yellow_half)
        self.score_circle = [(int((60 + i * 100) * self.ratio), self.margin_top//2, int(43 * self.ratio), False, False) for i in range(8)]

        # specific gibo
        self.eval = False


    def draw_sheet(self):    
        pygame.display.set_caption("2D Curling Simulator made by MILab (hufs)")
        pygame.display.set_icon(pygame.image.load('./GUI_utils/icon.png'))

        self.section_top = pygame.Rect(0, 0, self.nX, self.margin_top)
        self.section_mid = pygame.Rect(0, self.margin_top, self.nX, self.nY + self.margin_top)
        self.section_bottom = pygame.Rect(0, self.nY + self.margin_top, self.nX, self.margin_top + self.nY + self.margin_bottom)

        pygame.draw.rect(self.screen, (192, 192, 192), self.section_top)
        pygame.draw.rect(self.screen, (255, 255, 255), self.section_mid)
        pygame.draw.rect(self.screen, (192, 192, 192), self.section_bottom)

        # wheelchair line (horizontal) 
        pygame.draw.line(self.screen, (0, 0, 0), (self.hog_right, self.center_line - self.wheelchair_line_right), (self.tee_line_right, self.center_line - self.wheelchair_line_right))
        pygame.draw.line(self.screen, (0, 0, 0), (self.hog_right, self.center_line + self.wheelchair_line_right), (self.tee_line_right, self.center_line + self.wheelchair_line_right))

        # house (circle) 
        for i in range(3, -1, -1):   
            if i == 0 or i == 2:
                color = (255, 255, 255)
            elif i == 1:
                color = (203, 67, 53) # tomato
            else:
                color = (0, 102, 204) # royalblue
 
            pygame.draw.circle(self.screen, color, (self.tee_line_right, self.center_line), self.house_right[i])

        # center line (horizontal)
        pygame.draw.line(self.screen, (0, 0, 0), (0, self.center_line), (self.back_line_right + 30, self.center_line))

        # tee line (vertical) 
        pygame.draw.line(self.screen, (0, 0, 0), (self.tee_line_right, self.margin_top), (self.tee_line_right, self.margin_top + self.nY))

        # back line (vertical) 
        pygame.draw.line(self.screen, (0, 0, 0), (self.back_line_right, self.margin_top), (self.back_line_right, self.margin_top + self.nY), 2)

        # hog line (vertical)
        pygame.draw.line(self.screen, (255, 0, 0), (self.hog_left, self.margin_top), (self.hog_left, self.margin_top + self.nY), 3)
        pygame.draw.line(self.screen, (255, 0, 0), (self.hog_right, self.margin_top), (self.hog_right, self.margin_top + self.nY), 3) 

        self.update_turn_info()
        self.draw_text(self.screen, f'Team Red Score: {self.score_red}  Team Yellow Score: {self.score_yellow}', (self.nX - int(760 * self.ratio), int(76 * self.ratio)))

        if self.is_stop:
            self.draw_projected_path()


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

            
        if self.stones_fired_red + self.stones_fired_yellow <= 4:
            # 파이브 가드 확인
            first = True
            if self.current_team == 0:
                for idx, stone in enumerate(self.sm.stone_list_yellow):
                    if is_out(stone) and is_FGZ(idx, stone.radius, prev_stone_list_yellow):
                        if first:
                            self.sm.stone_list_red.pop()
                            first = False

                        stone.pos.x = prev_stone_list_yellow[idx][0]
                        stone.pos.y = prev_stone_list_yellow[idx][1]
            
            else:
                for idx, stone in enumerate(self.sm.stone_list_red):   
                    if is_out(stone) and is_FGZ(idx, stone.radius, prev_stone_list_red):
                        if first:
                            self.sm.stone_list_yellow.pop()
                            first = False

                        stone.pos.x = prev_stone_list_red[idx][0]
                        stone.pos.y = prev_stone_list_red[idx][1]


        for idx, stone in enumerate(self.sm.stone_list_red):
            if is_out(stone):
                self.sm.stone_list_red.pop(idx)

        for idx, stone in enumerate(self.sm.stone_list_yellow):
            if is_out(stone):
                self.sm.stone_list_yellow.pop(idx)

    
    def scoring(self):
        def distance(stone):
            return np.linalg.norm(np.array([stone.pos.x, stone.pos.y]) - np.array([self.tee_line_right, self.center_line]))

        red_in_house = [stone for stone in self.sm.stone_list_red if distance(stone) - stone.radius <= self.house_right[3]]
        yellow_in_house = [stone for stone in self.sm.stone_list_yellow if distance(stone) - stone.radius <= self.house_right[3]]


        all_stones = [(distance(stone), "red") for stone in red_in_house] + \
                 [(distance(stone), "yellow") for stone in yellow_in_house]
        all_stones.sort() 

        if len(all_stones) == 0:
            return 0, 0 
        
        score = 0
        closest_team = all_stones[0][1]
        for _, team in all_stones:
            if team == closest_team:
                score += 1
            else:
                break
        
        if closest_team == 'red':
            return score, 0
        else:
            return 0, score


    def get_stone(self):
        self.delta_P.x, self.delta_P.y = self.calculate_delta(self.slider_force.getValue(), self.slider_angle.getValue() - self.angle_range) 
        force = self.delta_P.magnitude() * self.scale 
        self.delta_P.normalize() 
        initCondVec = self.delta_P * force   

        stone = Stone(self.start_P.x, self.start_P.y, r=self.radius, c = (255, 0, 0))
        stone.vel.x, stone.vel.y = initCondVec.x, initCondVec.y    
        stone.initial_angleVel = self.slider_ang_vel.getValue() - 1.5
        stone.scale = self.scale 

        return stone


    def run(self): 
        running = True 
        self.reset()

        while running: 
            self.draw_sheet() 
            event_list = pygame.event.get()    
            pygame_widgets.update(event_list) 

            for event in event_list: 
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:  # 우클릭 (오른쪽 버튼)
                        print(f"우클릭 위치: {event.pos}")
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  
                        running = False 

                    if event.key == pygame.K_a: 
                        self.btn_auto()  

                    if event.key == pygame.K_r: 
                        self.btn_random()
                           
                    if event.key == pygame.K_s: 
                        self.btn_shot()


            self.draw_text(self.screen, f'force: {self.slider_force.getValue():.2f}', (int(180 * self.ratio), int(660 * self.ratio))) 
            self.draw_text(self.screen, f'angle: {self.slider_angle.getValue() - self.angle_range:.2f}°', (int(180 * self.ratio), int(710 * self.ratio))) 
            self.draw_text(self.screen, f'ang_vel: {self.slider_ang_vel.getValue() - 1.5:.2f}', (int(180 * self.ratio), int(760 * self.ratio)))

            if self.eval:
                self.sm.update(eval=True, current_team=self.current_team) 
            else:
                self.sm.update() 

            pygame.display.flip()  

        pygame.quit()
    
    
    def shot(self):
        def shot_thread(): 
            if self.current_team == 0: 
                self.stones_fired_red += 1
            else:
                self.stones_fired_yellow += 1

            print(f'\n\n ================ \n')
            print(f'[Team Red Fired - Team Yellow Fired] ▷▷▷ {self.stones_fired_red} - {self.stones_fired_yellow}') 
            print(f'[force] ▷▷▷ {self.slider_force.getValue()}')
            print(f'[angle] ▷▷▷ {self.slider_angle.getValue() - self.angle_range:.2f}°')
            print(f'[angle velocity] ▷▷▷ {self.slider_ang_vel.getValue() - 1.5}')

            # ################## 
            prev_stone_list_red, prev_stone_list_yellow = [], []

            for stone in self.sm.stone_list_red: 
                prev_stone_list_red.append([stone.pos.x, stone.pos.y])
                
            for stone in self.sm.stone_list_yellow: 
                prev_stone_list_yellow.append([stone.pos.x, stone.pos.y])

            self.stone = self.get_stone()
            self.sm.add(self.stone, self.current_team)
 
            self.is_stop = False # (공 정지 전까지 shot이 호출되는 것을 방지하기 위함)
            start = time.time()
            while True:
                all_stop = True
                time.sleep(1)

                for stone in (self.sm.stone_list_red + self.sm.stone_list_yellow):
                    if not(stone.vel.x == 0 and stone.vel.y == 0):
                        all_stop = False
                
                if all_stop:
                    # chek rule and penalty
                    end = time.time()
                    print(end-start)
                    break
 
            ### 공이 정지 후  
            self.check_out(prev_stone_list_red, prev_stone_list_yellow)
            self.score_red, self.score_yellow = self.scoring()  
            
            if self.stones_fired_red == 8 and self.stones_fired_yellow == 8:  
                self.is_reset = True

                time.sleep(2)
                self.reset()
        
            self.is_stop = True
            self.current_team = 1 - self.current_team 

        if not self.is_reset and self.is_stop:
            threading.Thread(target=shot_thread, daemon=True).start()



if __name__ == '__main__': 
    simulator = CurlingSimulator()
    simulator.run() 


