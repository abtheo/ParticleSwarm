import argparse
import sys
import pygame as pg
from pygame.locals import *
import numpy as np
from boid import Boid
from collections import deque   
import glob
import matplotlib.pyplot as plt
import random
from PIL import Image


def draw(screen, background, boids):
    """
    Draw things to the window. Called once per frame.
    """
    # Redraw screen here
    boids.clear(screen, background)
    dirty = boids.draw(screen)

    # Flip the display so that the things we drew actually show up.
    pg.display.update(dirty)

def main(target_img, num_boids=256, geometry="256x256", iterations=500):
    # Initialise pg.
    pg.init()

    # Set up the clock to maintain a relatively constant framerate.
    fps = 60.0
    fpsClock = pg.time.Clock()

    # Set up the window.
    window_width, window_height = [int(x) for x in geometry.split("x")]
    flags = DOUBLEBUF

    screen = pg.display.set_mode((window_width, window_height), flags)
    screen.set_alpha(None)
    background = pg.Surface((window_width, window_height)).convert()
    background.fill(pg.Color('white'))
    screen.fill(pg.Color('white'))
    pg.display.flip()

    boids = pg.sprite.RenderUpdates()

    add_boids(boids, num_boids, target_img)

    # Main game loop.
    dt = 1/fps  # dt is the time since last frame.

    min_mae = np.inf
    min_screen = []
    # while True:
    for t in range(iterations):
        events = pg.event.get() 
        #Iterate Boids and apply forces
        for b in boids:
            b.update(dt, boids)
        
        draw(screen, background, boids)
        #Get screenshot of game
        #Compare to Target image
        screenshot = pg.surfarray.array3d(pg.display.get_surface())
        tmask = (np.sum(target_img, axis=-1) < 225*3)
        mae = np.sum(np.absolute(target_img[tmask] - screenshot[tmask]))


        dt = fpsClock.tick(fps)
        if t < 20:
            continue
        # #Save lowest loss
        if mae < min_mae:
            min_mae = mae
            min_screen = screenshot


    return min_screen, min_mae


def add_boids(boids, num_boids, target_img):
    for _ in range(num_boids):
        boids.add(Boid(target_img=target_img))


if __name__ == "__main__":
    # random.seed(12)
    # random.shuffle(files)
    files = glob.glob("./pkmn_test/*.jpg")
    for i,fn in enumerate(files):
        target_image = pg.image.load(fn)
        target = pg.surfarray.array3d(target_image)

        out_name = f"./results/{i}.png"

        min_screen, mae = main(target, num_boids=400, geometry="256x256")
        print(f"File {fn}\nMinumum MAE: {mae}")
        min_screen = min_screen.swapaxes(0,1)
        with open("Log.txt", "a") as file_object:
            file_object.write(f"\n{fn}, {mae}, {out_name}")
        
        
        im = Image.fromarray(np.uint8(min_screen))
        im.save(out_name)
        
