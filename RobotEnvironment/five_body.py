import tkinter as tk
import math
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple

# Constants
WIDTH, HEIGHT = 800, 800
HEPTAGON_RADIUS = 300
BALL_RADIUS = 20
GRAVITY = 0.2
FRICTION = 0.99
RESTITUTION = 0.8
ROTATION_SPEED = 2 * math.pi / 5  # 360 degrees per 5 seconds
BALL_COLORS = [
    '#f8b862', '#f6ad49', '#f39800', '#f08300', '#ec6d51',
    '#ee7948', '#ed6d3d', '#ec6800', '#ec6800', '#ee7800',
    '#eb6238', '#ea5506', '#ea5506', '#eb6101', '#e49e61',
    '#e45e32', '#e17b34', '#dd7a56', '#db8449', '#d66a35'
]

@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    number: int
    color: str

class BouncingBallsSimulation:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
        self.canvas.pack()
        
        self.center_x = WIDTH // 2
        self.center_y = HEIGHT // 2
        self.heptagon_angle = 0
        self.heptagon_vertices = self._calculate_heptagon_vertices()
        
        # Create balls
        self.balls = []
        for i in range(20):
            self.balls.append(Ball(
                x=self.center_x,
                y=self.center_y,
                vx=random.uniform(-3, 3),
                vy=random.uniform(-3, 3),
                angle=0,
                angular_velocity=random.uniform(-0.1, 0.1),
                number=i+1,
                color=BALL_COLORS[i]
            ))
        
        self.last_time = 0
        self.running = True
        self.root.after(16, self.update)  # Start the animation loop (~60 FPS)
        
    def _calculate_heptagon_vertices(self) -> List[Tuple[float, float]]:
        """Calculate the vertices of the heptagon based on current rotation."""
        vertices = []
        for i in range(7):
            angle = self.heptagon_angle + 2 * math.pi * i / 7
            x = self.center_x + HEPTAGON_RADIUS * math.cos(angle)
            y = self.center_y + HEPTAGON_RADIUS * math.sin(angle)
            vertices.append((x, y))
        return vertices
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment."""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line vector
        line_vec = np.array([x2 - x1, y2 - y1])
        line_len = np.linalg.norm(line_vec)
        line_unit = line_vec / line_len
        
        # Point vector relative to line start
        point_vec = np.array([x - x1, y - y1])
        
        # Project point onto line
        proj = np.dot(point_vec, line_unit)
        
        if proj < 0:
            # Closest point is line start
            closest = line_start
        elif proj > line_len:
            # Closest point is line end
            closest = line_end
        else:
            # Closest point is along the line
            closest = (x1 + line_unit[0] * proj, y1 + line_unit[1] * proj)
        
        # Distance from point to closest point
        return math.sqrt((x - closest[0])**2 + (y - closest[1])**2)
    
    def _check_heptagon_collision(self, ball: Ball) -> bool:
        """Check if ball is colliding with heptagon walls."""
        for i in range(7):
            start = self.heptagon_vertices[i]
            end = self.heptagon_vertices[(i + 1) % 7]
            
            distance = self._point_to_line_distance((ball.x, ball.y), start, end)
            if distance < BALL_RADIUS:
                # Calculate normal vector of the wall
                wall_vec = np.array([end[0] - start[0], end[1] - start[1]])
                normal = np.array([-wall_vec[1], wall_vec[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Reflect velocity
                velocity = np.array([ball.vx, ball.vy])
                dot_product = np.dot(velocity, normal)
                reflected = velocity - 2 * dot_product * normal
                
                # Apply restitution and friction
                ball.vx = reflected[0] * RESTITUTION * FRICTION
                ball.vy = reflected[1] * RESTITUTION * FRICTION
                
                # Move ball outside the wall to prevent sticking
                penetration = BALL_RADIUS - distance
                ball.x += normal[0] * penetration
                ball.y += normal[1] * penetration
                
                # Add some angular velocity from the collision
                ball.angular_velocity += random.uniform(-0.5, 0.5) * (1 - FRICTION)
                
                return True
        return False
    
    def _check_ball_collisions(self):
        """Check for and resolve collisions between balls."""
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                ball1 = self.balls[i]
                ball2 = self.balls[j]
                
                dx = ball2.x - ball1.x
                dy = ball2.y - ball1.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 2 * BALL_RADIUS:
                    # Collision detected
                    nx = dx / distance
                    ny = dy / distance
                    
                    # Calculate relative velocity
                    dvx = ball2.vx - ball1.vx
                    dvy = ball2.vy - ball1.vy
                    
                    # Calculate impulse
                    impulse = (dvx * nx + dvy * ny) * RESTITUTION
                    
                    # Apply impulse
                    ball1.vx += impulse * nx * 0.5
                    ball1.vy += impulse * ny * 0.5
                    ball2.vx -= impulse * nx * 0.5
                    ball2.vy -= impulse * ny * 0.5
                    
                    # Move balls apart to prevent sticking
                    overlap = 2 * BALL_RADIUS - distance
                    ball1.x -= overlap * nx * 0.5
                    ball1.y -= overlap * ny * 0.5
                    ball2.x += overlap * nx * 0.5
                    ball2.y += overlap * ny * 0.5
                    
                    # Add some angular velocity from the collision
                    ball1.angular_velocity += (random.uniform(-0.5, 0.5) * (1 - FRICTION))
                    ball2.angular_velocity += (random.uniform(-0.5, 0.5) * (1 - FRICTION))
    
    def update(self):
        if not self.running:
            return
        
        # Clear canvas
        self.canvas.delete('all')
        
        # Update heptagon rotation
        self.heptagon_angle += ROTATION_SPEED / 60  # Assuming ~60 FPS
        self.heptagon_vertices = self._calculate_heptagon_vertices()
        
        # Draw heptagon
        self.canvas.create_polygon(
            *[coord for vertex in self.heptagon_vertices for coord in vertex],
            fill='', outline='black', width=2
        )
        
        # Update and draw balls
        for ball in self.balls:
            # Apply gravity
            ball.vy += GRAVITY
            
            # Apply friction
            ball.vx *= FRICTION
            ball.vy *= FRICTION
            ball.angular_velocity *= FRICTION
            
            # Update position
            ball.x += ball.vx
            ball.y += ball.vy
            ball.angle += ball.angular_velocity
            
            # Check collisions with heptagon walls
            self._check_heptagon_collision(ball)
            
            # Draw ball
            self.canvas.create_oval(
                ball.x - BALL_RADIUS, ball.y - BALL_RADIUS,
                ball.x + BALL_RADIUS, ball.y + BALL_RADIUS,
                fill=ball.color, outline='black'
            )
            
            # Draw number (rotated with the ball)
            text_x = ball.x + BALL_RADIUS * 0.6 * math.cos(ball.angle)
            text_y = ball.y + BALL_RADIUS * 0.6 * math.sin(ball.angle)
            self.canvas.create_text(
                text_x, text_y,
                text=str(ball.number),
                font=('Arial', int(BALL_RADIUS * 0.8)),
                fill='black'
            )
        
        # Check ball-ball collisions
        self._check_ball_collisions()
        
        # Schedule next update
        self.root.after(16, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Bouncing Balls in Spinning Heptagon")
    simulation = BouncingBallsSimulation(root)
    root.mainloop()