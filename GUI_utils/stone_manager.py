import numpy as np
import pygame
import time  

class StoneManager:

    def __init__(self, screen, opt):
        self.screen = screen 

        self.StoneList = []
        self.stone_list_red = []
        self.stone_list_yellow = [] 

        self.collision = False

        if opt == None:
            self.game_speed = 20
        else:
            self.game_speed = opt.game_speed

    
    def add(self, stone, current_team):
        if current_team == 0: # 팀 0일 때의 설정 
            stone.color = (240, 49, 38) 
            stone.team = 0 
            self.stone_list_red.append(stone)

        else: # 팀 1일 때의 설정 
            stone.color = (241, 196, 15)
            stone.team = 1 
            self.stone_list_yellow.append(stone)

        self.StoneList.append(stone)
 
    def apply_friction(self, stone1, stone2): # scale관련 넣어야할 수 있음
        # 접촉면 벡터 계산
        # friction_stone = 0.01 # ! 0.008 # e

        # friction 계산을 위한 벡터 - 두 stone사이의 접촉면의 방향을 나타냄.
        # 충돌방향 벡터
        contact_normal = (stone2.pos - stone1.pos) 
        contact_normal.normalize() # n
        # min_distance = stone1.radius + stone2.radius  # 두 스톤의 최소 거리 (겹침 방지)
    
        # distance = contact_normal.magnitude()
        # # 돌들이 겹쳤을 때 위치 조정
        # if distance < min_distance:
        #     overlap = min_distance - distance
        #     correction_vector = contact_normal * (overlap / distance / 2)
        #     stone1.pos += correction_vector
        #     stone2.pos -= correction_vector
        #     contact_normal.normalize()


        # 상대 속도 계산
        relative_velocity = (stone2.vel - stone1.vel) * stone1.scale
        # relative_velocity.normalize() # 이게 필요한가?

        # 상대 속도에서 접촉 방향의 성분 계산
        relative_speed_along_normal = relative_velocity.dotProduct(contact_normal)
        

        # 마찰력을 계산하고 적용
        if relative_speed_along_normal > 0:

            e = 0.7 # 탄성계수 - 빙판 고려
            # friction_force = contact_normal * friction_stone * relative_speed_along_normal * (-1.0)
            # friction_force = contact_normal * (friction_stone) * relative_speed_along_normal * (-0.5)
            j = (-(1+e)*relative_speed_along_normal) / (1 / stone1.mass + 1 / stone2.mass)
            impulse = contact_normal.__mul__(j / stone1.scale)


            stone1.vel = stone1.vel + impulse.__mul__(1/ stone1.mass * stone1.scale)# * (stone1.ratio)
            stone2.vel = stone2.vel - impulse.__mul__(1/ stone2.mass * stone2.scale)# * (stone2.ratio)
        
            #####3
            collision_point = stone1.pos + contact_normal * stone1.radius * stone1.scale
            r1 = collision_point - stone1.pos  # 충돌점으로부터 중심까지의 벡터
            r2 = collision_point - stone2.pos

            # 각속도 변화량 계산 (토크 = r x F / I, F는 충격량)
            torque1 = np.cross([r1.x, r1.y], [impulse.x, impulse.y]) / stone1.moment_of_inertia
            torque2 = np.cross([r2.x, r2.y], [impulse.x, impulse.y]) / stone2.moment_of_inertia

            # 각속도 업데이트
            stone1.angleVel += torque1
            stone2.angleVel -= torque2

        # stone1.angleVel = ((stone1.angleVel/2) - (stone2.angleVel/2.0)) ## ! 0.8
        # stone2.angleVel = ((stone2.angleVel/2) - (stone1.angleVel/2.0)) ## ! 0.8
 
 
    def update(self, eval=False, current_team=None): 
        # while True:
        for i in range(self.game_speed):
            n = len(self.stone_list_red) + len(self.stone_list_yellow)
            stoneList = self.stone_list_red + self.stone_list_yellow

            for i in range(0, n - 1):
                for j in range(i + 1, n):
                    if (stoneList[i].vel.x == 0.0 and stoneList[i].vel.y == 0.0) and (stoneList[j].vel.x == 0.0 and stoneList[j].vel.y == 0.0):
                        continue

                    colTest = stoneList[i].collide(stoneList[j])
                    if colTest:
                        self.collision = True 
                        self.apply_friction(stoneList[i], stoneList[j])

            all_stop = True

            for stone in (self.stone_list_red + self.stone_list_yellow):
                stone.update()  
                if not(stone.vel.x == 0 and stone.vel.y == 0):
                    all_stop = False
                # stone.drawTrajectory(self.screen)
            
            if all_stop == True:
                break
            
        for idx, stone in enumerate(self.stone_list_red + self.stone_list_yellow):
            stone.draw(self.screen) 

            if eval:  
                if current_team == 0:
                    if idx == len(self.stone_list_red) - 1: 
                        stone.drawTrajectory(self.screen, fire=True)
                    else: 
                        stone.drawTrajectory(self.screen)

                else:
                    if idx == len(self.StoneList) - 1: 
                        stone.drawTrajectory(self.screen, fire=True)
                    else: 
                        stone.drawTrajectory(self.screen)



            

