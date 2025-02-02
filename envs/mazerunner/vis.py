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
from mazerunner import MazeRunnerEnv

def visualize(env):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    fig, ax = plt.subplots(1,1,figsize=(5,5),constrained_layout=True)

    #plt.tight_layout()
    background = np.ones((env.maze_dim, env.maze_dim, 3), dtype=np.uint8)
    #maze_img = np.ones((env.maze_dim, env.maze_dim+1, 3), dtype=np.uint8)*220
    maze_img = (
        background * abs(np.expand_dims(env.maze, -1) - 1) * 255
    )  # zero out (white) where there is a valid path

    
    #for i, p in enumerate(prompt):
    #    maze_img[p[0], p[1], :] = prompt_color
    #maze_img[env.pos[0], env.pos[1], :] = [110, 110, 110]  # grey
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.imshow(maze_img)

    # task goal
    x,y = env.goal_positions[-1]
    ax.add_patch(patches.Rectangle((y-0.46, x-0.47), 0.9, 0.9, color="red"))
    plt.text(y, x, 'T', ha="center", va="center", color='white', fontsize=12)
    # start pos
    x,y = env.start
    ax.add_patch(patches.Rectangle((y-0.46, x-0.47), 0.9, 0.9, color="red"))
    plt.text(y, x, 'S', ha="center", va="center", color='white', fontsize=12)



if __name__=='__main__':
    env = MazeRunnerEnv(maze_dim=30, min_num_goals=1, max_num_goals=1)
    env.reset()
    visualize(env)
    plt.savefig('maze.png', dpi=500)