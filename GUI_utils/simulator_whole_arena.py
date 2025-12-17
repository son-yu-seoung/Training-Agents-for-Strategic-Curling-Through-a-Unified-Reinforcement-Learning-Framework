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



class CurlingSimulator(Buttons, Utils):
    def __init__(self, opt = None):
        self.opt = opt # from agent
        self.init_scale(self.opt)
        
        pygame.init() 
        pygame.font.init()
        self.font = pygame.font.SysFont("arial", int(40 * self.ratio), True, True)

        self.screen = pygame.display.set_mode((self.nX, self.nY + self.margin_top + self.margin_bottom))
        self.screen.fill((255, 255, 255))

        # 현재까지의 code 정리 및 stone의 크기는 미리 정의
        # self.sm = StoneManager()
        # self.sm.seScreen()

        self.init_variables(eval=False)
        self.init_widgets()

    
    def init_scale(self, opt): 
        if opt == None:
            self.ratio = 0.4
        else:
            self.ratio = opt.ratio

        self.nX = 4572 * self.ratio # 45.72m 
        self.nY = 475 * self.ratio # 5m
        
        self.center_x = self.nX / 2
        self.center_y= self.nY / 2

        self.margin_top = 140 * self.ratio
        self.margin_bottom = 300 * self.ratio
        self.center_line = self.margin_top + (self.nY // 2)

        self.tee = self.center_x - (1737.5 * self.ratio) # 17.375m
        self.back_line = self.tee - (182.9 * self.ratio) # 1.829m
        self.hog_line = self.tee + (640.1 * self.ratio) # 6.401m 
        self.center_line_y = self.margin_top + (self.nY // 2)
        self.hack_line = 45.7 / 2 * self.ratio # 0.457m (총)
        self.wheelchair_line = 45.7 * self.ratio 
        self.house_r = [15.2 * self.ratio, 61 * self.ratio, 121.9 * self.ratio, 182.9 * self.ratio] 
         

        # house
        self.house_right = [15.2 * self.ratio, 61 * self.ratio, 121.9 * self.ratio, 182.9 * self.ratio] 

        # score circle (x, y, r, red_half, yellow_half)
        self.score_circle = [(int((60 + i * 100) * self.ratio), self.margin_top//2, int(43 * self.ratio), False, False) for i in range(8)]


    def init_widgets(self):
        handle_radius = round(20 * self.ratio)
        self.slider_force = Slider(self.screen, int(150 * self.ratio), int(840 * self.ratio), int(250 * self.ratio), int(40 * self.ratio), min=0, max=8, step=0.001, colour=(255, 255, 255), handleRadius=handle_radius) 
        self.slider_angle = Slider(self.screen, int(20 * self.ratio), int(680 * self.ratio), int(40 * self.ratio), int(180 * self.ratio), min=0, max=2 * self.angle_range, colour=(255, 255, 255), handleRadius=handle_radius, step=0.01, vertical=True)   
        self.slider_ang_vel = Slider(self.screen, int(80 * self.ratio), int(680 * self.ratio), int(40 * self.ratio), int(180 * self.ratio), min=0, max=3 , colour=(255, 255, 255), handleRadius=handle_radius, step=0.01, vertical=True)   

        self.button_auto = Button(self.screen, int(440 * self.ratio), int(630 * self.ratio), int(140 * self.ratio), int(80 * self.ratio), text='auto', fontSize=int(50 * self.ratio), onClick=self.btn_auto)
        self.button_random = Button(self.screen, int(440 * self.ratio), int(725 * self.ratio), int(140 * self.ratio), int(80 * self.ratio), text='random', fontSize=int(50 * self.ratio), onClick=self.btn_random)
        self.button_shot = Button(self.screen, int(440 * self.ratio), int(820 * self.ratio), int(140 * self.ratio), int(80 * self.ratio), text='shot', fontSize=int(50 * self.ratio), onClick=self.btn_shot)
 
    def init_variables(self, eval):
        if not eval: # eval일 때는 current_team 초기화 X
            self.current_team = 0

        self.stones_fired_red = 0
        self.stones_fired_yellow = 0
        self.score_red = 0
        self.score_yellow = 0

        self.angle_range = self.calculate_angle((self.hog_line, self.center_line), (self.nX - self.hog_line, self.margin_top))

        self.is_reset = False
        self.is_stop = True 
        self.is_auto = False
        self.state = False

        self.sm = StoneManager(self.screen, self.opt) 

        self.start_P = Point(self.hog_line, self.center_line)
        self.delta_P = Point() 

        self.radius = 13.9 * self.ratio
        self.stones = []
        self.scale = (self.nX - self.hog_line - self.hog_line) / 50.0 

    def draw_sheet(self):    
        pygame.display.set_caption("2D Curling Simulator made by MILab (hufs)")
        pygame.display.set_icon(pygame.image.load('./GUI_utils/icon.png'))

        self.section_top = pygame.Rect(0, 0, self.nX, self.margin_top)
        self.section_mid = pygame.Rect(0, self.margin_top, self.nX, self.nY + self.margin_top)
        self.section_bottom = pygame.Rect(0, self.nY + self.margin_top, self.nX, self.margin_top + self.nY + self.margin_bottom)

        pygame.draw.rect(self.screen, (192, 192,192), self.section_top)
        pygame.draw.rect(self.screen, (255, 255, 255), self.section_mid)
        pygame.draw.rect(self.screen, (192,192,192), self.section_bottom)

        # wheelchair line (horizontal)
        pygame.draw.line(self.screen, (0, 0, 0), (self.tee, self.center_line_y - self.wheelchair_line), (self.hog_line, self.center_line_y - self.wheelchair_line))
        pygame.draw.line(self.screen, (0, 0, 0), (self.tee, self.center_line_y + self.wheelchair_line), (self.hog_line, self.center_line_y + self.wheelchair_line))
        pygame.draw.line(self.screen, (0, 0, 0), (self.nX - self.hog_line, self.center_line_y - self.wheelchair_line), (self.nX - self.tee, self.center_line_y - self.wheelchair_line))
        pygame.draw.line(self.screen, (0, 0, 0), (self.nX - self.hog_line, self.center_line_y + self.wheelchair_line), (self.nX - self.tee, self.center_line_y + self.wheelchair_line))

        # house (circle) 
        for i in range(3, -1, -1):   
            if i == 0 or i == 2:
                color = (255, 255, 255)
            elif i == 1:
                color = (203, 67, 53) # tomato
            else:
                color = (0, 102, 204) # royalblue

            pygame.draw.circle(self.screen, color, (self.tee, self.center_line_y), self.house_r[i]) 
            pygame.draw.circle(self.screen, color, (self.nX - self.tee, self.center_line_y), self.house_r[i])

        # center line (horizontal)
        pygame.draw.line(self.screen, (0, 0, 0), (182.9 * self.ratio, self.center_line_y), (self.nX - (182.9 * self.ratio), self.center_line_y))

        # tee line (vertical)
        pygame.draw.line(self.screen, (0, 0, 0), (self.tee, self.margin_top), (self.tee, self.margin_top + self.nY))
        pygame.draw.line(self.screen, (0, 0, 0), (self.nX - self.tee, self.margin_top), (self.nX - self.tee, self.margin_top + self.nY))

        # back line (vertical)
        pygame.draw.line(self.screen, (0, 0, 0), (self.back_line - 1, self.margin_top), (self.back_line - 1, self.margin_top + self.nY), 2)
        pygame.draw.line(self.screen, (0, 0, 0), (self.nX - self.back_line - 1, self.margin_top), (self.nX - self.back_line - 1, self.margin_top + self.nY), 2)

        # hack line (vertical)
        pygame.draw.line(self.screen, (0, 0, 0), (182.9 * self.ratio, self.center_line_y - self.hack_line), (182.9 * self.ratio, self.center_line_y + self.hack_line))
        pygame.draw.line(self.screen, (0, 0, 0), (self.nX - (182.9 * self.ratio), self.center_line_y - self.hack_line), (self.nX - (182.9 * self.ratio), self.center_line_y + self.hack_line))
        
        # hog line (vertical)
        pygame.draw.line(self.screen, (255, 0, 0), (self.hog_line, self.margin_top), (self.hog_line, self.margin_top + self.nY), 3)
        pygame.draw.line(self.screen, (255, 0, 0), (self.nX - self.hog_line, self.margin_top), (self.nX - self.hog_line, self.margin_top + self.nY), 3)
        self.update_turn_info()
        self.draw_text(self.screen, f'Team Red Score: {self.score_red}  Team Yellow Score: {self.score_yellow}', (self.nX - int(760 * self.ratio), int(76 * self.ratio)))

        if self.is_stop:
            self.draw_projected_path()
        print('house', (self.nX - self.tee, self.center_line_y))
        print('tee', self.nX - self.tee)
        print('back', self.nX - self.back_line - 1)
        print('hog left', self.hog_line)
        print('hog right', self.nX - self.hog_line)

        

    def init_back_end(self):
        pass
        # self.sm = StoneManager()

    
    def run(self):
        pygame.init()  
        running = True
        self.draw_sheet() 

        while running:  
            event_list = pygame.event.get()  
            
            pygame_widgets.update(event_list)  

            for event in event_list:
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        # 왼쪽 화살표 키를 누르면 스톤의 가속도를 증가시킴
                        print("end")
                        running = False

            self.draw_text(self.screen, f'force: {self.slider_force.getValue():.2f}', (int(180 * self.ratio), int(660 * self.ratio))) 
            self.draw_text(self.screen, f'angle: {self.slider_angle.getValue() - self.angle_range:.2f}°', (int(180 * self.ratio), int(710 * self.ratio))) 
            self.draw_text(self.screen, f'ang_vel: {self.slider_ang_vel.getValue() - 1.5:.2f}', (int(180 * self.ratio), int(760 * self.ratio)))

            pygame.display.flip()  

        pygame.quit()



if __name__ == '__main__': 
    simulator = CurlingSimulator()
    simulator.run() 



'''

1. pygame init
2. static variance define
3. main loop (사용자 입력 처리, 게임 상태 업데이트, 게임 상태 표시시)
  clock.tick(10) # fps'
  pygame.display.update()

4. pygmae quit


'''