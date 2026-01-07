from GUI_utils.point import Point 
import pygame 
import numpy as np


class Circle:

    def __init__(self, x=0, y=0, r=10, c=(255, 0, 0)):

        self.pos = Point(x, y)
        self.vel = Point(0, 0) # vec + mag
        self.acc = Point(0, 0) # 가속도
        
        self.radius = r
        self.color = c

        self.default_ratio = 0.27
        self.ratio = 0.7
        self.speed_ratio = self.ratio/self.default_ratio
        self.scale = 0.0 # stone 만들때 설정필요

        # 스톤 관련 변수
        self.mass = 18.0 # 17.24 ~ 19.96 kg
        self.gravity = 9.81
        self.k = 0.1               # Magnus 계수
        self.dt = 0.0005           # 시간 간격 (s)
        self.c = 0.5               # 속도 비례 마찰 계수 (임의의 값)
        self.beta = 0.1            # 회전 감속 계수
        self.initial_mu = 0.02     # 초기 마찰 계수
        self.alpha = 0.05          # 마찰 감속 계수

        self.initial_angleVel = 0
        self.angleVel = -2 ## 회전속도 # -2
        self.angleAcc = 0.0
        self.t = 0

    def __str__(self):
        return f"Circle({self.pos.x}, {self.pos.y})"


    def update(self):
        
        # 현재 속도 계산 
        v = self.vel.magnitude() 
        
        if v < 0.01 * self.scale:
        # if np.abs(self.vel.x) < 0.001 * self.speed_ratio and np.abs(self.vel.y) < 0.001 * self.speed_ratio:
            self.vel.x, self.vel.y = 0.0, 0.0
            self.angleVel = 0.0
            self.t = 0
            # print('나 거의 멈춰가')
        else:
            mu = self.initial_mu * np.exp(-self.alpha * self.t)
            self.angleVel = self.initial_angleVel * np.exp(-self.beta * self.t)

            friction_force = mu * self.mass * self.gravity * self.scale + self.c * v
            friction_acc = friction_force / self.mass

            # Magnus 효과에 의한 힘 계산
            magnus_force_x = self.k * self.angleVel * self.vel.y
            magnus_force_y = self.k * self.angleVel * self.vel.x

            # 가속도 계산 
            self.acc.x = ((-friction_acc * self.vel.x / v) + (magnus_force_x / self.mass))
            self.acc.y = ((-friction_acc * self.vel.y / v) + (magnus_force_y / self.mass))

            # 속도 업데이트
            self.vel = self.vel + self.acc * self.dt

            # 위치 업데이트
            self.pos = self.pos + self.vel * self.dt 
            self.t += self.dt
        
    def collide(self, other):
        diff = self.pos - other.pos
        length = diff.magnitude()

        if length <= (self.radius + other.radius):
            return True
        else:
            return False

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.pos.x, self.pos.y), self.radius)


