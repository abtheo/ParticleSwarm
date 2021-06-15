import pygame as pg
from random import uniform
from vehicle import Vehicle
import numpy as np
from scipy import signal
import random

class Boid(Vehicle):

    # CONFIG
    k=5
    debug = False
    min_speed = .001
    max_speed = .05
    max_force = 1
    max_turn = 360
    perception = 4
    can_wrap = False
    edge_distance_pct = 5
    ###############

    def __init__(self, target_img):
        Boid.set_boundary(Boid.edge_distance_pct)

        # Randomize starting position and velocity
        start_position = pg.math.Vector2(
            uniform(0, Boid.max_x),
            uniform(0, Boid.max_y))
        
        start_velocity = pg.math.Vector2(
            uniform(-1, 1) * Boid.max_speed,
            uniform(-1, 1) * Boid.max_speed)

        super().__init__(start_position, start_velocity,
                         Boid.min_speed, Boid.max_speed,
                         Boid.max_force, Boid.can_wrap)

        self.rect = self.image.get_rect(center=self.position)

        self.debug = Boid.debug

        self.target_img = target_img

        #Calculate Force kernel -
        #(x,y) distance from origin for each cells
        Mf = []
        Imf = []
        for i in range(-self.k//2+1,self.k//2+1):
            row = []
            row2 = []
            for j in range(-self.k//2+1,self.k//2+1):
                row.append([j,i])
                row2.append((i**2 + j**2)**0.5)
            Mf.append(row)
            Imf.append(row2)

        self.force_kernel = np.array(Mf)
        #Calculate Distance Kernel -
        #Euclidean of each (x,y) in the Force Kernel
        self.distance_kernel = np.array(Imf)
        mask = (self.distance_kernel != 0)
        self.distance_kernel[mask] = np.divide(1,self.distance_kernel[mask])**2
        self.distance_kernel = np.expand_dims(self.distance_kernel, axis=-1)

    def separation(self, boids):
        steering = pg.Vector2()
        for boid in boids:
            dist = self.position.distance_to(boid.position)
            if dist < self.perception:
                steering -= boid.position - self.position
        steering = self.clamp_force(steering)
        return steering

    # def alignment(self, boids):
    #     steering = pg.Vector2()
    #     for boid in boids:
    #         steering += boid.velocity
    #     steering /= len(boids)
    #     steering -= self.velocity
    #     steering = self.clamp_force(steering)
    #     return steering / 8

    # def cohesion(self, boids):
    #     steering = pg.Vector2()
    #     for boid in boids:
    #         steering += boid.position
    #     steering /= len(boids)
    #     steering -= self.position
    #     steering = self.clamp_force(steering)
    #     return steering / 1000

    def safe_edge(self, edge, axis=0):
        return int(max(0, min(self.target_img.shape[axis], edge)))

    def color_search(self):
        k = self.k #shorthand
        
        #Find kernel positions in image
        xmin = self.safe_edge(self.position.x-k//2)
        xmax = max(xmin+1, self.safe_edge(self.position.x+k//2+1))
        ymin = self.safe_edge(self.position.y-k//2,axis=1)
        ymax = max(ymin+1, self.safe_edge(self.position.y+k//2+1,axis=1))

        #Extract kernel
        kernel = self.target_img[xmin:xmax, ymin:ymax]
        #Shape mismatch, kernel cut-off by boundary
        if not kernel.shape[0] == k or not kernel.shape[1] == k:
            return pg.Vector2(), self.color

        center = kernel[kernel.shape[0]//2, kernel.shape[1]//2]

        color = np.array([self.color.r, self.color.g, self.color.b])

        # center_old = kernel[k//2,k//2]
        center_color = pg.Color(int(center[0]), int(center[1]), int(center[2]), 255)
        if center_color == self.color and not np.sum(center_color) > 225*3:
            # return pg.Vector2(), self.color
            return -self.velocity*0.001, self.color
        
        color_kernel = np.linalg.norm(kernel-color,axis=-1)
        mask = (color_kernel != 0)
        #Find euclidean distance in colorspace
        color_kernel[mask] = np.divide(1,color_kernel[mask])**2
        color_kernel = np.expand_dims(color_kernel, axis=-1)
        #Mean (Force * Color * Dist)
        vector_kernel = color_kernel*self.force_kernel*self.distance_kernel
        mean_force = np.mean(vector_kernel, axis=(0,1))
        return pg.Vector2(x=mean_force[0],y=mean_force[1]), center_color

    def update(self, dt, boids):
        steering = pg.Vector2()

        if not self.can_wrap:
            steering += self.avoid_edge()


        separation = self.separation(boids)
        # alignment = self.alignment(neighbors)
        # cohesion = self.cohesion(neighbors)

        steering += separation #+ cohesion #+ alignment
        vel, color = self.color_search()
        steering += vel
        new_direction = self.clamp_force(self.velocity + vel)
        steering = self.clamp_force(steering)

        # color = random.choice([pg.Color('green'), pg.Color('blue'), pg.Color('red')])
        super().update(dt, steering, new_direction, color)

    def get_neighbors(self, boids):
        neighbors = []
        for boid in boids:
            if boid != self:
                dist = self.position.distance_to(boid.position)
                if dist < self.perception:
                    neighbors.append(boid)
        return neighbors
