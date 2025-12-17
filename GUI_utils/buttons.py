import threading
import time
import random

class Buttons:
    def btn_auto(self): # 나중에 utils.py로 옮기기 모두
        if self.is_auto:
            self.is_auto = False
        else:
            self.is_auto = True
        
        def btn_auto_thread(): 
            while True:
                if self.is_auto == False:
                    break

                if self.stones_fired_red + self.stones_fired_yellow == 16:
                    break

                self.btn_random()
                self.btn_shot()
 
                time.sleep(1)
 
        threading.Thread(target=btn_auto_thread, daemon=True).start() 

    
    def btn_random(self):
        if self.is_stop:
            rand_force = round(random.uniform(0, 8), 2) 
            rand_angle = round(random.uniform(0, 2 * self.angle_range), 2)  
            rand_ang_vel = round(random.uniform(0, 3), 2)  

            self.slider_force.setValue(rand_force)
            self.slider_angle.setValue(rand_angle) 
            self.slider_ang_vel.setValue(rand_ang_vel) 


    def btn_shot(self):
        self.shot() 


    def are_stones_stopped(self): 
        for stone in (self.sm.stone_list_red + self.sm.stone_list_yellow):
            if not(stone.vel.x == 0 and stone.vel.y == 0):  
                return False  # 하나라도 움직이면 False 반환
        return True  # 모든 스톤이 정지한 경우 True 반환