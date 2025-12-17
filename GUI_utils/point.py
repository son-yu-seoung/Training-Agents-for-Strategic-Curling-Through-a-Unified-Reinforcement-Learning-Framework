import math

class Point:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def magnitude(self):
        if self.x == 0 and self.y == 0:
            self.x += 0.01
            self.y += 0.01

        elif self.x == 0:
            self.x += 0.01
        
        elif self.y == 0:
            self.y += 0.01
            
        return math.sqrt(self.x*self.x + self.y*self.y)

    def normalize(self):
        length = self.magnitude()

        if self.x == 0 and self.y == 0:
            self.x += 0.01
            self.y += 0.01

        elif self.x == 0:
            self.x += 0.01
        
        elif self.y == 0:
            self.y += 0.01
 
        self.x = self.x/length
        self.y = self.y/length


    def dotProduct(self, other):
        dot = float(self.x * other.x + self.y * other.y)

        mag = 0
        mag1 = self.magnitude()
        mag2 = other.magnitude() 

        if mag1 == 0 or mag2 == 0:
            mag = max(mag1, mag2) 
        else:
            mag = mag1 * mag2

        cos_theta = max(-1, min(1, dot / (mag + 1e-16)))
        theta = math.acos(cos_theta)
        return theta

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)

    def rotate(self, theta):
        # radian = math.radians(theta)
        radian = theta

        new_x = self.x * math.cos(radian) - self.y * math.sin(radian)
        new_y = self.x * math.sin(radian) + self.y * math.cos(radian)
        return Point(new_x, new_y)