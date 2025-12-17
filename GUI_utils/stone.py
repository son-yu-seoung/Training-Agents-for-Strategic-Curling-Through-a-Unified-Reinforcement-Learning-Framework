import pygame
from GUI_utils.circle import Circle
from GUI_utils.point import Point 

import os
import csv
import time
import numpy as np

class Stone(Circle):
 
    def __init__(self, x=0, y=0, r=3.475, c=(255, 0, 0)):

        super().__init__(x, y, r, c)  

        self.team = 0

        self.spinVector = Point(0, 1) #vector, center = (pos.x, pos.y)
        self.theta = 0
 
        self.trajectory = []
        self.trajectory.append(self.pos)

        # self.position_history = []
        self.moment_of_inertia = 0.5 * self.mass * (self.radius ** 2)

        self.prev_pos = np.array([self.pos.x, self.pos.y])
        self.color = c
 

    def drawGrip(self, screen):
        self.grip = self.spinVector * (self.radius * 0.7) 
        self.endPoint = (self.grip + self.pos)
        self.startPoint = (self.grip * -1.0 + self.pos) # 그립벡터의 시작점을 나타냄.
        pygame.draw.line(screen, (0, 0, 0), (self.startPoint.x, self.startPoint.y), 
                         (self.endPoint.x, self.endPoint.y), 3) ## 5

    def collide(self, other):
        diff = self.pos - other.pos
        length = diff.magnitude()

        if length <= (self.radius + other.radius):
            return True
        else:
            return False

    def update(self):
        super().update() # t 변화함
        
        n = len(self.trajectory)
        if (self.trajectory[n-1] - self.pos).magnitude() < 1.0:
            pass
        else:
            self.trajectory.append(self.pos)

        # self.position_history.append(Point(self.pos.x, self.pos.y))

        # self.theta = (self.theta + self.angleVel) #% 360
        # self.theta = self.theta * 0.9993 ## 0.9993
        # self.spinVector = Point(0, 1).rotate(self.angleVel) 
        self.spinVector = self.spinVector.rotate(self.angleVel * self.dt)



    def draw(self, screen):
        pygame.draw.circle(screen, (150, 150, 150), (self.pos.x, self.pos.y), self.radius)
        pygame.draw.circle(screen, self.color, (self.pos.x, self.pos.y), self.radius*0.8)  
        self.drawGrip(screen)
        

    def drawTrajectory(self, screen, fire=False): 
        if len(self.trajectory) < 8:
            pass
        else:
            for pos in self.trajectory:  
                if fire:
                    pygame.draw.circle(screen, (0, 255, 127), (pos.x, pos.y), self.radius*0.4) 
                else: 
                    pygame.draw.circle(screen, (255, 0, 255), (pos.x, pos.y), self.radius*0.4)


                    
                # if self.team == 0:
                #     pygame.draw.circle(screen, (255, 0, 255), (pos.x, pos.y), self.radius*0.4)
                #     # pygame.draw.circle(screen, (255, 102, 102), (pos.x, pos.y), self.radius*0.4)
                # else:
                #     pygame.draw.circle(screen, (0, 255, 127), (pos.x, pos.y), self.radius*0.4)
                #     # pygame.draw.circle(screen, (255, 178, 102), (pos.x, pos.y), self.radius*0.4)


