import gymnasium as gym
import numpy as np
from functools import partial
import copy
import random
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from env import *

class Env(KitchenToyEnv):
    def rend(self):
        fig, ax = plt.subplots(1,1,figsize=(5,5),constrained_layout=True)

        background = np.ones((10,10, 3), dtype=np.uint8)
        maze_img = background * 255

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['top'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.spines['right'].set_visible(False)

        done_color = [192,192,192]
        for i, s in enumerate(self.obj_states):
            x, y = OBJ_POSITIONS[i]
            if s==OBJ_GOAL_STATE:
                maze_img[x,y,:] = done_color
            else:
                maze_img[x,y,:] = OBJ_COLORS[i]

        maze_img[self.pos[0], self.pos[1], :] = [90, 90, 90]  # agent:black
        plt.imshow(maze_img)

        for i in range(N_OBJ):
            y, x = OBJ_POSITIONS[i]
            plt.text(x, y, str(i), ha="center", va="center",  fontsize=18)



if __name__=='__main__':
    env = Env()
    env.reset()
    env.rend()
    plt.savefig('gridworld.png', dpi=500)