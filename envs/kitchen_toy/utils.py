import numpy as np
import pickle
from .env import KitchenToyEnv
from copy import deepcopy
import os
import torch
import matplotlib.pyplot as plt


# make envs for eval with the tasks in trajs
def get_env_list(trajs, device, max_ep_len=90):
    infos, env_list = [], []
    for traj in trajs:
        env = KitchenToyEnv(task=traj['task'])
        info = {'max_ep_len': max_ep_len, 'state_dim': env.observation_space.shape[0],
                'act_dim': env.action_space.n, 'device': device, 'prompt_dim': env.prompt_dim,
                'discrete_action': env.discrete_action}
        infos.append(info)
        env_list.append(env)
    return infos, env_list

# load training dataset, and envs of some training/test tasks
def get_train_test_dataset_envs(dataset_path, device, max_ep_len, **kwargs):
    train_dataset_path = "{}_{}.pkl".format(dataset_path, 'train')
    test_dataset_path = "{}_{}.pkl".format(dataset_path, 'test')
    with open(train_dataset_path, 'rb') as f:
        trajectories_list = pickle.load(f)
    with open(test_dataset_path, 'rb') as f:
        test_trajectories_list = pickle.load(f)
    n_test = len(test_trajectories_list)
    val_trajectories_list = trajectories_list[0:n_test] # seen tasks to eval
    
    info, env_list = get_env_list(val_trajectories_list, device, max_ep_len)
    test_info, test_env_list = get_env_list(test_trajectories_list, device, max_ep_len)
    #print(test_info, test_env_list, info, env_list, len(trajectories_list))
    return info, env_list, val_trajectories_list, test_info, test_env_list, test_trajectories_list, trajectories_list


# given a trajectory, sample a goal-sequence as prompt
# Updated 1-19: the prompt becomes (g1,...,gk,T), where g is onehot and T is the multi-hot task description
# thus, the max_prompt_length is increased to 8
def get_prompt(trajectory, max_prompt_length=8, prompt_length=None, device=None, use_optimal_prompt=False):
    if prompt_length is None:
        # sample a prompt length between [1,max_prompt_length] for training
        prompt_length = np.random.randint(1,max_prompt_length+1)
    
    goal_timesteps = []
    if prompt_length > len(trajectory['prompts']): # if prompt_length exceeds optimal prompts
        prompt_length = len(trajectory['prompts'])
    if prompt_length>1:
        goal_range = np.arange(0, len(trajectory['prompts'])-1) # sample subgoals before the last task desc
        goal_timesteps = np.random.choice(goal_range, prompt_length-1, replace=False).tolist()
        goal_timesteps.sort()
    goal_timesteps.append(len(trajectory['prompts'])-1)

    # padding to the left
    #goal_timesteps = goal_timesteps + [len(trajectory['prompts'])-1]*(max_prompt_length-prompt_length)
    mask = np.concatenate([np.zeros((1, max_prompt_length - prompt_length)), np.ones((1, prompt_length))], axis=1)
    #print(goal_timesteps, mask)
    prompt = []
    for t in goal_timesteps:
        prompt.append(trajectory['prompts'][t])
    prompt = np.array(prompt).reshape(1,-1,7)
    prompt = np.concatenate([np.zeros((1, max_prompt_length - prompt_length, 7)), prompt], axis=1)
    prompt = torch.from_numpy(prompt).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(device=device)
    #print(prompt, goal_timesteps, mask)
    return prompt, mask


# compute the upperbound return for each task specified in the trajectory 
# (to compare with different methods). Input a list of trajectories, return a list 
# of return (float) of the task in each trajectory.
def get_oracle_returns(trajectories):
    return [1. for i in range(len(trajectories))]


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
        get_train_test_dataset_envs('kitchen_toy_t90', device, max_ep_len=90)
    #for i in range(100):
    get_prompt(trajectories_list[0], device=device)
    print(len(env_list), len(test_env_list), len(trajectories_list))
    #variant={'K':50, 'batch_size': 16, 'max_prompt_len': 5}
    #fn = get_prompt_batch(trajectories_list, info[0], variant)
    #prompt, batch = fn()
