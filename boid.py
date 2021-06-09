import pygame as pg
from random import uniform
from vehicle import Vehicle
import numpy as np
from scipy import signal

class Boid(Vehicle):

    # CONFIG
    debug = False
    min_speed = .01
    max_speed = .2
    max_force = 1
    max_turn = 12
    perception = 60
    crowding = 30
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

        self.k=25
        
        # self.force_kernel = np.array([ 
        #     [i,j]
        #     for i in range(-self.k//2+1,self.k//2+1)
        #     for j in range(-self.k//2+1,self.k//2+1)]) / (self.k//2-1)

        thing = []
        for i in range(-self.k//2+1,self.k//2+1):
            row = []
            for j in range(-self.k//2+1,self.k//2+1):
                row.append([j,i])
            thing.append(row)
        self.force_kernel = np.array(thing)

    def gkern(self, std=3):
        gkern1d = signal.gaussian(self.k, std=std).reshape(self.k, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d

    def separation(self, boids):
        steering = pg.Vector2()
        for boid in boids:
            dist = self.position.distance_to(boid.position)
            if dist < self.crowding:
                steering -= boid.position - self.position
        steering = self.clamp_force(steering)
        return steering

    def alignment(self, boids):
        steering = pg.Vector2()
        for boid in boids:
            steering += boid.velocity
        steering /= len(boids)
        steering -= self.velocity
        steering = self.clamp_force(steering)
        return steering / 8

    def cohesion(self, boids):
        steering = pg.Vector2()
        for boid in boids:
            steering += boid.position
        steering /= len(boids)
        steering -= self.position
        steering = self.clamp_force(steering)
        return steering / 100

    def safe_edge(self, edge, axis=0):
        return int(max(0, min(self.target_img.shape[axis], edge)))

    def kernel_search(self):
        k = self.k #shorthand

        # self.target_img
        kernel = self.target_img[self.safe_edge(self.position.x-k//2):self.safe_edge(self.position.x+k//2+1), 
                                self.safe_edge(self.position.y-k//2,axis=1):self.safe_edge(self.position.y+k//2+1,axis=1)]

        bool_kernel = np.any(kernel==255, axis=-1)
        #Shape mismatch, kernel cut-off by boundary
        if not bool_kernel.shape[0] == k or not bool_kernel.shape[1] == k:
            # return pg.Vector2()
            return 1

        # net_force = pg.Vector2()
        # if np.sum(bool_kernel) == 0:
        #     return pg.Vector2()
        return 1 - np.sum(bool_kernel) / (k*k)
            # return False
            # return pg.Vector2()

        # return pg.math.Vector2(
        #     uniform(-1, 1) * Boid.max_speed,
        #     uniform(-1, 1) * Boid.max_speed)

        # for force in self.force_kernel[bool_kernel]:
        #     net_force += force
        # return net_force/np.sum(bool_kernel)

    def ksearch(self):
        k = self.k #shorthand

        # self.target_img
        kernel = self.target_img[self.safe_edge(self.position.x-k//2):self.safe_edge(self.position.x+k//2+1), 
                                self.safe_edge(self.position.y-k//2,axis=1):self.safe_edge(self.position.y+k//2+1,axis=1)]

        bool_kernel = np.any(kernel==255, axis=-1).swapaxes(0,1)
        #Shape mismatch, kernel cut-off by boundary
        if not bool_kernel.shape[0] == k or not bool_kernel.shape[1] == k:
            # return pg.Vector2()
            return pg.Vector2()

        if not bool_kernel.any():
            return pg.Vector2()

        net_force = pg.Vector2()

        for force in self.force_kernel[bool_kernel]:
            net_force += force

        mean_force = np.mean(self.force_kernel[bool_kernel], axis=0)
        return pg.Vector2(x=mean_force[0],y=mean_force[1])


    def update(self, dt, boids, energy):
        steering = pg.Vector2()

        if not self.can_wrap:
            steering += self.avoid_edge()

        neighbors = self.get_neighbors(boids)
        if neighbors:
            separation = self.separation(neighbors)
            # alignment = self.alignment(neighbors)
            # cohesion = self.cohesion(neighbors)

            # DEBUG
            # separation *= 0
            # alignment *= 0
            # cohesion *= 0

            steering += separation #+ alignment# + cohesion

        steering = self.clamp_force(steering)

        jj = self.ksearch()
        if not jj.x == 0:
            dd = jj.as_polar()
            self.velocity.from_polar(dd)

        # steering += jj
        # if in_center:
        #     steering = self.velocity * -0.1

        super().update(dt, steering, energy)

    def get_neighbors(self, boids):
        neighbors = []
        for boid in boids:
            if boid != self:
                dist = self.position.distance_to(boid.position)
                if dist < self.perception:
                    neighbors.append(boid)
        return neighbors
