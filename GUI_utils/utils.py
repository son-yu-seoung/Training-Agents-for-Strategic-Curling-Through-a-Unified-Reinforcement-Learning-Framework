import pygame
import math

class Utils:
    def draw_text(self, screen, msg, pos, color=(0, 0, 0)):
        text = self.font.render(msg, True, color)
        screen.blit(text, pos)


    def update_turn_info(self):
        for n_shot in range(min(self.stones_fired_red, self.stones_fired_yellow)):
            x, y, r, red_half, yellow_half = self.score_circle[n_shot]
            red_half = True
            yellow_half = True
            self.score_circle[n_shot] = (x, y, r, red_half, yellow_half)
        

        if self.stones_fired_red > self.stones_fired_yellow:
            x, y, r, red_half, yellow_half = self.score_circle[self.stones_fired_red - 1]
            red_half = True 
            self.score_circle[self.stones_fired_red - 1] = (x, y, r, red_half, yellow_half)
        

        elif self.stones_fired_red < self.stones_fired_yellow:
            x, y, r, red_half, yellow_half = self.score_circle[self.stones_fired_yellow - 1]
            yellow_half = True
            self.score_circle[self.stones_fired_yellow - 1] = (x, y, r, red_half, yellow_half)

        self.draw_turn_info()

    
    def draw_turn_info(self): 
        for x, y, r, red_half, yellow_half in self.score_circle:
            pygame.draw.circle(self.screen, (0,0,0), (x, y), r, 3)
            
            if red_half: 
                self.draw_half_circle(self.screen, (255, 0, 0), (x, y), r - 3, 180, 360)

            if yellow_half: 
                self.draw_half_circle(self.screen, (255, 255, 0), (x, y), r - 3, 0, 180)
        
        self.draw_text(self.screen, f'Team Red Fired: {self.stones_fired_red}/8  Team Yellow Fired: {self.stones_fired_yellow}/8', (self.nX - int(760 * self.ratio), int(16 * self.ratio)))

    
    def draw_half_circle(self, surface, color, center, radius, start_angle, end_angle):
        points = [center]
        for angle in range(start_angle, end_angle + 1):
            rad = math.radians(angle)
            points.append((center[0] + int(radius * math.cos(rad)), center[1] + int(radius * math.sin(rad))))
        pygame.draw.polygon(surface, color, points)

    
    def calculate_angle(self, p1, p2): 
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) 

        # 아크탄젠트 함수로 각도 계산
        angle_rad = math.atan(slope)  

        # 라디안을 각도로 변환
        angle_deg = math.degrees(angle_rad)

        return abs(int(angle_deg))
    

    def calculate_delta(self, force, angle):
        radian = math.radians(angle)
        x2 = self.start_P.x + force * math.cos(radian)
        y2 = self.start_P.y - force * math.sin(radian)

        return x2 - self.start_P.x, y2 - self.start_P.y


    def draw_projected_path(self, c=(0, 206, 209), arrow_size=16): 
        arrow_size *= self.ratio
        delta_x, delta_y = self.calculate_delta(self.slider_force.getValue() * 10, self.slider_angle.getValue() - self.angle_range)
 
        start = [self.start_P.x, self.start_P.y]
        end = [start[0] + delta_x, start[1] + delta_y]

        pygame.draw.line(self.screen, c, (start[0], start[1]), (end[0], end[1]), int(arrow_size//1.2))
        rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
        pygame.draw.polygon(self.screen, c, ((end[0]+arrow_size*math.sin(math.radians(rotation)), end[1]+arrow_size*math.cos(math.radians(rotation))), (end[0]+arrow_size*math.sin(math.radians(rotation-120)), end[1]+arrow_size*math.cos(math.radians(rotation-120))), (end[0]+arrow_size*math.sin(math.radians(rotation+120)), end[1]+arrow_size*math.cos(math.radians(rotation+120)))))