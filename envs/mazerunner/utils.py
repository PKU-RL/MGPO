import numpy as np
import pickle
from .mazerunner import MazeRunnerEnv
from copy import deepcopy
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

'''
For online evaluation: MazeRunner env with a fixed maze and goal across episodes
'''
class MazeRunnerEvalEnv(MazeRunnerEnv):
    def __init__(self, maze_dim, min_num_goals, max_num_goals, maze, goal_pos):
        self.env_name = 'mazerunner'
        self.maze = deepcopy(maze)
        self.goal_positions = deepcopy(goal_pos)
        self.prompt_dim = 2 # goal: (x,y)
        super().__init__(maze_dim, min_num_goals, max_num_goals)
        self.discrete_action = True if hasattr(self.action_space, 'n') else False
    
    def reset(self, *args, **kwargs):
        self.start = (self.maze_dim - 2, self.maze_dim // 2)
        empty_locations = [x for x in zip(*np.where(self.maze == 0))]
        empty_locations.remove(self.start)
        self.active_goal_idx = 0
        self.pos = self.start
        self._enforce_reset = False
        self._plotting = False
        self._goal_render_texts = [None for _ in range(len(self.goal_positions))]
        self.episode_visited_goals = [] # save prompt goals in episode experience, for online adaptation
        return self._get_obs()

    def step(self, act):
        obs, rew, terminated, truncated, info = super().step(act)
        self.episode_visited_goals.append(np.asarray(obs[0:2])) # save prompt goals in episode experience, for online adaptation
        return obs, rew, terminated, info

    def render_with_prompt(self, prompt, return_rgb=False):
        if not self._plotting:
            self.start_plotting()
            plt.ion()
            self._plotting = True

        plt.tight_layout()
        background = np.ones((self.maze_dim, self.maze_dim, 3), dtype=np.uint8)
        maze_img = (
            background * abs(np.expand_dims(self.maze, -1) - 1) * 255
        )  # zero out (white) where there is a valid path

        prompt_color = [240, 3, 252]
        for p in prompt:
            maze_img[p[0], p[1], :] = prompt_color

        maze_img[self.pos[0], self.pos[1], :] = [110, 110, 110]  # grey
        plt.imshow(maze_img)

        for i, p in enumerate(prompt):
            plt.text(
                p[1], p[0], str(i), ha="center", va="center"
            )

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

    def get_task_prompt(self):
        return [np.array(p)/self.maze_dim for p in self.goal_positions]

# make envs for eval with the tasks in trajs
def get_env_list(trajs, device, max_ep_len=50):
    infos, env_list = [], []
    for traj in trajs:
        maze = traj['maze']
        goal_pos = traj['goal_pos']
        env = MazeRunnerEvalEnv(maze_dim=maze.shape[0], min_num_goals=1, 
                                max_num_goals=len(goal_pos), maze=maze, goal_pos=goal_pos)
        info = {'max_ep_len': max_ep_len, 'state_dim': env.observation_space.shape[0],
                'act_dim': env.action_space.n, 'device': device, 'prompt_dim': env.prompt_dim,
                'discrete_action': env.discrete_action}
        infos.append(info)
        env_list.append(env)
    return infos, env_list

# load training dataset, and envs of some training/test tasks
def get_train_test_dataset_envs(dataset_path, device, max_ep_len=50, n_train_env=50, n_test_env=50, **kwargs):
    fn = os.path.basename(dataset_path)
    #max_ep_len = int(fn.split('-')[-2][1:])
    #print(max_ep_len)

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    test_trajectories_list = trajectories[0:n_test_env] # unseen tasks to eval
    trajectories_list = trajectories[n_test_env:] # offline data for training
    val_trajectories_list = trajectories_list[0:n_train_env] # seen tasks to eval
    
    info, env_list = get_env_list(val_trajectories_list, device, max_ep_len)
    test_info, test_env_list = get_env_list(test_trajectories_list, device, max_ep_len)
    #print(test_info, test_env_list, info, env_list, len(trajectories_list))
    return info, env_list, val_trajectories_list, test_info, test_env_list, test_trajectories_list, trajectories_list


# given a trajectory, sample a goal-sequence as prompt
def get_prompt(trajectory, max_prompt_length=5, prompt_length=None, device=None, use_optimal_prompt=False):
    if prompt_length is None:
        # sample a prompt length between [1,max_prompt_length] for training
        prompt_length = np.random.randint(1,max_prompt_length+1)
    
    goal_timesteps = []
    if use_optimal_prompt and 'optimal_prompts' in trajectory: # use the saved optimal prompt
        if prompt_length > trajectory['optimal_prompts'].shape[0]-1: # if prompt_length exceeds optimal prompts
            prompt_length = trajectory['optimal_prompts'].shape[0]-1
        if prompt_length>1:
            goal_range = np.arange(1, trajectory['optimal_prompts'].shape[0]-1)
            goal_timesteps = np.random.choice(goal_range, prompt_length-1, replace=False).tolist()
            goal_timesteps.sort()
        goal_timesteps.append(trajectory['optimal_prompts'].shape[0]-1)
    else: # sample (len-1) goals + the last goal in the trajectory as the prompt
        if prompt_length>1:
            goal_range = np.arange(1, trajectory['timesteps']-1)
            prompt_length = min(prompt_length, trajectory['timesteps']-1) # prompt len cannot exceed traj len
            #print('pl:', prompt_length)
            goal_timesteps = np.random.choice(goal_range, prompt_length-1, replace=False).tolist()
            goal_timesteps.sort()
        goal_timesteps.append(trajectory['timesteps']-1) # the last goal in the prompt is the task-goal
    #print(prompt_length, goal_timesteps)

    # padding to the left
    goal_timesteps = [0]*(max_prompt_length-prompt_length) + goal_timesteps # pad with the initial position
    mask = np.concatenate([np.zeros((1, max_prompt_length - prompt_length)), np.ones((1, prompt_length))], axis=1)
    #print(goal_timesteps, mask)

    prompt = []
    for t in goal_timesteps:
        if use_optimal_prompt and 'optimal_prompts' in trajectory:
            prompt.append(trajectory['optimal_prompts'][t])
        else:
            #print(trajectory['next_observations'][t][0:2]*15, trajectory['goal_pos'])
            prompt.append(trajectory['next_observations'][t][0:2])
    prompt = np.array(prompt).reshape(1,-1,2)

    prompt = torch.from_numpy(prompt).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(device=device)
    #print(prompt, goal_timesteps, mask)
    return prompt, mask


# compute the upperbound return for each task specified in the trajectory 
# (to compare with different methods). Input a list of trajectories, return a list 
# of return (float) of the task in each trajectory.
def get_oracle_returns(trajectories, reward_step_penalty=-0.1):
    ret = []
    for t in trajectories:
        l = len(t['optimal_prompts'])
        ret.append(1+l*reward_step_penalty)
    return ret


if __name__=='__main__':
    '''
    with open('envs/mazerunner/mazerunner-d15-g1-astar.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    main_keys = ['observations', 'next_observations', 'actions', 'rewards', 'terminals']
    info_keys = ['timesteps', 'maze', 'goal_pos']
    print(len(trajectories))

    for k in main_keys:
        print(k, type(trajectories[0][k]), trajectories[0][k].dtype, trajectories[0][k].shape)

    for k in info_keys:
        print(k, trajectories[0][k])
    '''
    import torch
    device=torch.device('cuda')
    info, env_list, val_trajectories_list, test_info, test_env_list, test_trajectories_list, trajectories_list = \
        get_train_test_dataset_envs('mazerunner-d30-g4-4-t500-multigoal-astar.pkl', device, max_ep_len=500)
    #for i in range(100):
    #    get_prompt(trajectories_list[0], device=device)
    #variant={'K':50, 'batch_size': 16, 'max_prompt_len': 5}
    #fn = get_prompt_batch(trajectories_list, info[0], variant)
    #prompt, batch = fn()

    for env, traj in zip(test_env_list, test_trajectories_list):
        env.reset()
        prompts = (traj['optimal_prompts']*env.maze_dim).astype(int)
        for p in prompts:
            env.pos = p            
            env.render()
