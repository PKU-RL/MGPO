import numpy as np
import pickle
from .kitchen_seq_wrapper import KitchenSeqEnv, N_GOALS_TEST
from copy import deepcopy
import os
import torch
import matplotlib.pyplot as plt
from gymnasium import spaces

# make envs for eval with the tasks in trajs
def get_env_list(trajs, device, max_ep_len, render_mode=None):
    infos, env_list = [], []
    for traj in trajs:
        env = KitchenSeqEnv(task=traj['task'], render_mode=render_mode, object_noise_ratio=0, robot_noise_ratio=0)
        #print(env.observation_space, env.action_space)
        info = {'max_ep_len': max_ep_len, 'state_dim': env.observation_space['observation'].shape[0],
                'act_dim': env.action_space.shape[0], 'device': device, 'prompt_dim': env.prompt_dim,
                'discrete_action': env.discrete_action}
        infos.append(info)
        env_list.append(env)
    return infos, env_list

# load training dataset, and envs of some training/test tasks
def get_train_test_dataset_envs(dataset_path, device, max_ep_len, render_mode=None, only_test=False, **kwargs):
    train_dataset_path = "{}_{}.pkl".format(dataset_path, 'train')
    test_dataset_path = "{}_{}.pkl".format(dataset_path, 'test')
    if only_test:
        trajectories_list = None
    else:
        with open(train_dataset_path, 'rb') as f:
            trajectories_list = pickle.load(f)
    with open(test_dataset_path, 'rb') as f:
        test_trajectories_list = pickle.load(f)
    n_test = len(test_trajectories_list)
    val_trajectories_list, info, env_list = [], [], [] 
    test_info, test_env_list = get_env_list(test_trajectories_list, device, max_ep_len, render_mode=render_mode)
    #print(test_info, test_env_list, info, env_list, len(trajectories_list))
    return info, env_list, val_trajectories_list, test_info, test_env_list, test_trajectories_list, trajectories_list


# given a trajectory, sample a goal-sequence as prompt
# the prompt is (g1,...,gk,T), where g is onehot and T is the multi-hot task description
# thus, the max_prompt_length is 5+1=6
def get_prompt(trajectory, max_prompt_length=6, prompt_length=None, device=None, use_optimal_prompt=False):
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
    prompt = np.array(prompt).reshape(1,-1,N_GOALS_TEST)
    prompt = np.concatenate([np.zeros((1, max_prompt_length - prompt_length, N_GOALS_TEST)), prompt], axis=1)
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

    def obs_to_qpos_qvel(obs):
        robot_pos, robot_vel, obj_pos, obj_vel = obs[0:9], obs[9:18], obs[18:39], obs[39:]
        qpos, qvel = np.concatenate((robot_pos, obj_pos)), np.concatenate((robot_vel, obj_vel))
        return qpos, qvel

    import torch
    device=torch.device('cuda')
    info, env_list, val_trajectories_list, test_info, test_env_list, test_trajectories_list, trajectories_list = \
        get_train_test_dataset_envs('kitchen_t500', device, max_ep_len=500)
    
    for env, traj in zip(env_list[0:], val_trajectories_list[0:]):
        print(traj['prompts'], env.subtask_names)
        for o in traj['observations']:
            q_pos, q_vel = obs_to_qpos_qvel(o)
            env.robot_env.set_state(q_pos, q_vel)
            env.render()