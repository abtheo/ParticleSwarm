# Import standard modules.
import argparse
import sys

# Import non-standard modules.
import pygame as pg
from pygame.locals import *
import numpy as np

# Import local modules
from boid import Boid
from collections import deque
from sklearn.preprocessing import MinMaxScaler

default_boids = 50
default_geometry = "1000x1000"

target_image = pg.image.load("Circle.png")
target = pg.surfarray.array3d(target_image)

def draw(screen, background, boids):
    """
    Draw things to the window. Called once per frame.
    """

    # Redraw screen here
    boids.clear(screen, background)
    dirty = boids.draw(screen)

    # Flip the display so that the things we drew actually show up.
    pg.display.update(dirty)

def main(args):
    # Initialise pg.
    pg.init()
    # pg.event.set_allowed([pg.QUIT, pg.KEYDOWN, pg.KEYUP])

    # Set up the clock to maintain a relatively constant framerate.
    fps = 60.0
    fpsClock = pg.time.Clock()

    # Set up the window.
    window_width, window_height = [int(x) for x in args.geometry.split("x")]
    flags = DOUBLEBUF

    screen = pg.display.set_mode((window_width, window_height), flags)
    screen.set_alpha(None)
    background = pg.Surface(screen.get_size()).convert()
    background.fill(pg.Color('black'))

    boids = pg.sprite.RenderUpdates()

    add_boids(boids, args.num_boids)

    # Main game loop.
    dt = 1/fps  # dt is the time since last frame.

    energy = 1.0
    maxlen = 3000
    mae_history = deque(maxlen=maxlen)
    mae_history.extend([765000000.0]*maxlen)
    while True:
        #Iterate Boids and apply forces
        for b in boids:
            b.update(dt, boids, energy)
        
        draw(screen, background, boids)
        #Get screenshot of game
        #Compare to Target image
        # screenshot = pg.surfarray.array3d(pg.display.get_surface())
        # mae = np.sum(np.absolute(target - screenshot))
        # mae_history.append(mae)

        # #Normalize to 0-1
        # energy = (mae - min(mae_history)) / (max(mae_history) - min(mae_history))

        # print(f"MAE: {mae}, Energy: {energy}")
        

        dt = fpsClock.tick(fps)


def add_boids(boids, num_boids):
    for _ in range(num_boids):
        boids.add(Boid(target_img=target))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emergent flocking.')
    parser.add_argument('--geometry', metavar='WxH', type=str,
                        default=default_geometry, help='geometry of window')
    parser.add_argument('--number', dest='num_boids', default=default_boids,
                        help='number of boids to generate')
    args = parser.parse_args()

    main(args)
