'''
A toy environment, mirroring the Kitchen sequential tasks
7 objects are scattered in the 2D gridworld, each represents a subtask target. 
A task specifies sequential execution of a subset of subtasks. 
The agent should visit and manipulate them in order
'''
import gymnasium as gym
import numpy as np
from functools import partial
import copy
import random
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

MAZE_DIM=10
N_OBJ = 7
OBJ_INIT_STATE = 0
OBJ_GOAL_STATE = 1
OBJ_POSITIONS = [[9, 1],[8, 4],[1, 9],[3, 7],[3, 9],[6, 1],[8, 6]]
OBJ_COLORS = [[255,192,203], [255,255,0], [30,144,255], [189,252,201], [221,160,221], [237,145,33], [176,224,230]]


class KitchenToyEnv(gym.Env):
    def __init__(self, task=None):
        self.env_name = 'kitchen_toy'
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(9,)
        ) # obj states + agent pos
        self.action_space = gym.spaces.Discrete(6) # move + manipulate
        self.discrete_action = True
        self.prompt_dim = N_OBJ # goal obj_states

        # make a maze as an n x n array
        self.maze = np.ones((MAZE_DIM,MAZE_DIM), dtype=int)*-1
        for i, p in enumerate(OBJ_POSITIONS):
            self.maze[p[0], p[1]] = i

        if task is not None:
            self.task = task
            self.num_subtasks = len(task)
        else:
            self.sample_task()
        self.max_return = self.num_subtasks
        self.reset()

        self._plotting = False
        self._fig, self._ax = None, None

    def sample_task(self, num_subtasks=None):
        if num_subtasks is None:
            num_subtasks = random.randint(3, N_OBJ)
        self.num_subtasks = num_subtasks
        self.max_return = self.num_subtasks
        self.task = random.sample([i for i in range(N_OBJ)], k=num_subtasks)
        return self.task

    # get the whole prompt sequence
    def get_optimal_prompt(self):
        '''
        prompt = []
        completion = [0 for i in range(N_OBJ)]
        for i in range(self.num_subtasks):
            completion[self.task[i]] = 1
            prompt.append(np.array(completion, dtype=np.float32))
        return prompt
        '''
        # Updated 1-19: the whole prompt is (g1, ..., gk, T)
        # subgoals g are onehot for manipulated object, while T is the multi-hot task goal
        prompt = []
        for i in range(self.num_subtasks):
            subgoal = np.zeros(N_OBJ, dtype=np.float32)
            subgoal[self.task[i]] = 1
            prompt.append(subgoal)
        prompt += self.get_task_prompt()
        return prompt

    # get the final goal of the task
    def get_task_prompt(self):
        completion = [0 for i in range(N_OBJ)]
        for i in range(self.num_subtasks):
            completion[self.task[i]] = 1
        return [np.array(completion, dtype=np.float32)]

    def print_episode_stats(self, prompt=None):
        prompt_list=[]
        if prompt is not None:
            prompt_np = (prompt[0].cpu().numpy()[0]).astype(int)
            mask_np = prompt[1].cpu().numpy()[0].astype(int)
            prompt_np = prompt_np[(mask_np!=0).argmax():]
            prompt_list = []
            for i, p in enumerate(prompt_np):
                if i<len(prompt_np)-1:
                    prompt_list.append(np.argmax(p))
                else:
                    prompt_list.append(p)
        print('task: {}, prompt: {}, visited: {}'.format(self.task, prompt_list, self.episode_visited_goal_ids))

    def reset(self):
        self.start = (np.random.randint(0,MAZE_DIM), np.random.randint(0,MAZE_DIM))
        self.pos = self.start
        self.obj_states = [OBJ_INIT_STATE for i in range(N_OBJ)]
        
        self.active_goal_idx = 0
        self._enforce_reset = False
        self._plotting = False
        self.episode_visited_goals, self.episode_visited_goal_ids = [], [] # save prompt goals in episode experience, for online adaptation
        return self._get_obs()

    def step(self, act):
        assert not self._enforce_reset, "Reset the environment with `env.reset()`"
        # 0 --> west, 1 --> north, 2 --> east, 3 --> south, 4 --> none, 5 --> manipulate
        if act==5: # flip the obj state
            obj = self.maze[self.pos[0], self.pos[1]]
            if obj>=0:
                if self.obj_states[obj]==OBJ_INIT_STATE:
                    self.obj_states[obj] = OBJ_GOAL_STATE
                    subgoal = np.zeros(N_OBJ, dtype=np.float32)
                    subgoal[obj] = 1
                    if obj not in self.episode_visited_goal_ids:
                        self.episode_visited_goals.append(subgoal) # if an obj goal state is triggered, add to visited goals
                        self.episode_visited_goal_ids.append(obj)
                else:
                    self.obj_states[obj] = OBJ_INIT_STATE
        else:
            dirs = [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]
            chosen_dir = np.array(dirs[act])
            desired_loc = tuple(self.pos + chosen_dir)

            valid = True
            for coord in desired_loc:
                if coord < 0 or coord >= MAZE_DIM:
                    valid = False
            if valid:
                self.pos = desired_loc

        success = self.detect_success() #self.obj_states[self.task[self.active_goal_idx]] == OBJ_GOAL_STATE
        terminated = False
        rew = float(success) / self.max_return # normalized score as reward
        obs = self._get_obs()

        if success:
            if self.active_goal_idx == self.num_subtasks - 1:
                terminated = True
                self._enforce_reset = True
            self.active_goal_idx += 1

        return obs, rew, terminated, {"success": success}

    # subtask success: subtasks before current should be done; subtasks after current should not be done
    def detect_success(self):
        success = True
        for i in range(self.active_goal_idx+1):
            if self.obj_states[self.task[i]] != OBJ_GOAL_STATE:
                success = False
                break
        for i in range(self.active_goal_idx+1, self.num_subtasks):
            if self.obj_states[self.task[i]] == OBJ_GOAL_STATE:
                success = False
                break
        return success

    def _get_obs(self):
        i, j = tuple(self.pos)
        obs = np.array(self.obj_states + [i,j], dtype=np.float32)
        return obs

    def render(self, return_rgb=False):
        if not self._plotting:
            self.start_plotting()
            plt.ion()
            self._plotting = True

        plt.tight_layout()
        background = np.ones((MAZE_DIM, MAZE_DIM, 3), dtype=np.uint8)
        maze_img = background * 255

        done_color = [192,192,192]
        for i, s in enumerate(self.obj_states):
            x, y = OBJ_POSITIONS[i]
            if s==OBJ_GOAL_STATE:
                maze_img[x,y,:] = done_color
            else:
                maze_img[x,y,:] = OBJ_COLORS[i]

        maze_img[self.pos[0], self.pos[1], :] = [0, 0, 0]  # agent:black
        plt.imshow(maze_img)

        for i in range(N_OBJ):
            y, x = OBJ_POSITIONS[i]
            plt.text(x, y, str(i), ha="center", va="center")

        plt.draw()
        plt.pause(0.1)

        if return_rgb:
            canvas = FigureCanvasAgg(plt.gcf())
            canvas.draw()
            w, h = canvas.get_width_height()
            buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (w,h,4)
            buf = np.roll(buf, 3, axis=2)
            image = Image.frombytes("RGBA", (w,h), buf.tostring())
            image = np.asarray(image)[:,:,:3]
            return image

    def start_plotting(self):
        if self._fig:
            plt.close()
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)

def expert_policy(env):
    gx, gy = OBJ_POSITIONS[env.task[env.active_goal_idx]]
    px, py = env.pos
    if gx==px and gy==py:
        return 5
    elif gx==px:
        if gy<py:
            return 0
        else:
            return 2
    elif gy==py:
        if gx<px:
            return 1
        else:
            return 3
    else:
        r = random.randint(0,1)
        if r==0:
            if gx<px:
                return 1
            else:
                return 3
        else:
            if gy<py:
                return 0
            else:
                return 2

from itertools import permutations
def enum_tasks():
    objs = [i for i in range(N_OBJ)]
    tasks = []
    for n in range(3, N_OBJ+1):
        for p in permutations(objs, n):
            tasks.append(list(p))
    #print(tasks)
    return tasks


if __name__=='__main__':
    '''
    env = KitchenToyEnv()
    
    for ep in range(10):
        env.sample_task()
        print(env.task)
        print(env.get_prompt())
        env.reset()
        for t in range(100):
            env.render()
            #act = env.action_space.sample()
            act = expert_policy(env)
            s, r, d, info = env.step(act)
    
            #print(s, r, d)
            #input()
            if d:
                break
    '''
    t = enum_tasks()
    print(t)
    print(len(t))