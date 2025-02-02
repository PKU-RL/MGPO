'''
prompt utils for training & evaluation
'''

import numpy as np
#import gym
import json, pickle, random, os, torch
from collections import namedtuple
#from .prompt_evaluate_episodes import prompt_evaluate_episode, prompt_evaluate_episode_rtg
import random
from copy import deepcopy

''' sample batch from trajectories dataset '''

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

# given a trajectory, convert it into Transformer sequence
def get_sequence(trajectory, K, info):
    state_dim, act_dim, device, discrete_action = info['state_dim'], info['act_dim'], info['device'], info['discrete_action']
    max_len = K

    s = trajectory['observations'].reshape(1, -1, state_dim)
    if discrete_action:
        a = trajectory['actions'].reshape(1, -1)
    else:
        a = trajectory['actions'].reshape(1, -1, act_dim)
    r = trajectory['rewards'].reshape(1, -1, 1)
    d = trajectory['terminals'].reshape(1, -1)
    timesteps = np.arange(0, trajectory['timesteps']).reshape(1, -1)
    rtg = discount_cumsum(trajectory['rewards'], gamma=1.)[:s.shape[1] + 1].reshape(1, -1, 1)
    if rtg.shape[1] <= s.shape[1]:
        rtg = np.concatenate([rtg, np.zeros((1, 1, 1))], axis=1)
    #print(s.shape, a, r, d, timesteps)

    # padding to the right
    tlen = s.shape[1]
    s = np.concatenate([s, np.zeros((1, max_len - tlen, state_dim))], axis=1)
    if discrete_action:
        a = np.concatenate([a, np.zeros((1, max_len - tlen))], axis=1)
    else:
        a = np.concatenate([a, np.zeros((1, max_len - tlen, act_dim))], axis=1)
    r = np.concatenate([r, np.zeros((1, max_len - tlen, 1))], axis=1)
    d = np.concatenate([d, np.ones((1, max_len - tlen)) * 2], axis=1)
    rtg = np.concatenate([rtg, np.zeros((1, max_len - tlen, 1))], axis=1)
    timesteps = np.concatenate([timesteps, np.zeros((1, max_len - tlen))], axis=1)
    mask = np.concatenate([np.ones((1, tlen)), np.zeros((1, max_len - tlen))], axis=1)

    s = torch.from_numpy(s).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(a).to(dtype=torch.long if discrete_action else torch.float32, device=device)
    r = torch.from_numpy(r).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(d).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(mask).to(device=device)
    #print(s.shape, a, timesteps, mask, tlen, trajectory['timesteps'])

    return s, a, r, d, rtg, timesteps, mask


# from the dataset, sample a batch of prompt-sequence pairs for training
def get_prompt_batch(trajectories, info, variant, get_prompt):
    num_trajectories = len(trajectories)
    max_ep_len, max_prompt_len = variant['max_ep_len'], variant['max_prompt_len']
    state_dim, act_dim, device, discrete_action = info['state_dim'], info['act_dim'], info['device'], info['discrete_action']
    batch_size, K = variant['batch_size'], variant['K']
    #print(discrete_action, batch_size, device, act_dim, K, num_trajectories)
    subsample, subsample_minlen = variant['subsample_trajectory'], variant['subsample_min_len']

    def fn(batch_size=variant['batch_size']):
        # sample batch indices in the trajectories dataset
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True
        )
        prompt_list, p_mask_list = [], []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []
        
        for i in (batch_inds):
            if subsample:
                trajectory = subsample_trajectory(trajectories[i], subsample_minlen)
            else:
                trajectory = trajectories[i]
            #print(trajectory['timesteps'], trajectory['observations'].shape)
            p, p_mask = get_prompt(trajectory, max_prompt_length=max_prompt_len, device=device, use_optimal_prompt=False)
            prompt_list.append(p)
            p_mask_list.append(p_mask)

            s, a, r, d, rtg, timesteps, mask = get_sequence(trajectory, K, info)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)
            rtg_list.append(rtg)
            timesteps_list.append(timesteps)
            mask_list.append(mask)

        p, p_mask = torch.cat(prompt_list, dim=0), torch.cat(p_mask_list, dim=0)
        s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list, dim=0)
        rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)
        prompt = p, p_mask
        batch = s, a, r, d, rtg, timesteps, mask
        return prompt, batch
    
    return fn


# subsample into traj[0:len], len in [minlen, traj len]
def subsample_trajectory(trajectory, minlen):
    l = random.randint(minlen, trajectory['timesteps'])
    if l==trajectory['timesteps']:
        return trajectory
    keys = ['observations', 'actions', 'rewards', 'terminals', 'next_observations']
    new_trajectory = {}
    for k in trajectory:
        if k=='timesteps':
            new_trajectory[k] = l 
        elif k in keys:
            new_trajectory[k] = deepcopy(trajectory[k][0:l])
        else:
            new_trajectory[k] = deepcopy(trajectory[k])
    return new_trajectory


""" evaluation """

def eval_episodes(info, variant, envs, model, prompt_len, get_prompt, trajectories):
    max_ep_len, discrete_action = info['max_ep_len'], info['discrete_action']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']

    returns = []
    ep_lens = []
    #norm_scores = []
    for env_id in range(len(envs)):
        prompt = get_prompt(trajectories[env_id], variant['max_prompt_len'], 
                            prompt_length=prompt_len, device=device, use_optimal_prompt=variant['test_optimal_prompt'])

        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret, ep_len = prompt_evaluate_episode(
                    envs[env_id],
                    state_dim,
                    act_dim,
                    discrete_action,
                    model,
                    max_ep_len=max_ep_len,
                    device=device,
                    prompt=prompt,
                    )
                #print(prompt, ret)
            returns.append(ret)
            ep_lens.append(ep_len)
            #if hasattr(envs[env_id], 'max_return'):
            #    norm_scores.append(ret/envs[env_id].max_return)
    ret = {
        f'prompt_len_{prompt_len}_return_mean': np.mean(returns),
        #f'prompt_len_{prompt_len}_return_std': np.std(returns),
        f'prompt_len_{prompt_len}_ep_len_mean': np.mean(ep_lens),
        }
    #if len(norm_scores)>0:
    #    ret[f'prompt_len_{prompt_len}_normalized_score_mean'] = np.mean(norm_scores)
    return ret


def prompt_evaluate_episode(
        env,
        state_dim,
        act_dim,
        discrete_action,
        model,
        max_ep_len=1000,
        device='cuda',
        prompt=None
    ):

    model.eval()
    model.to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    if discrete_action:
        actions = torch.zeros((0,), device=device, dtype=torch.long)
    else:
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    #rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # print('evaluate/t', t)
        # add padding
        if discrete_action:
            actions = torch.cat([actions, torch.zeros((1,), device=device, dtype=torch.long)], dim=0)
        else:
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        #rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        #print(states.shape, actions, timesteps, prompt)
        action = model.get_action(
            states.to(dtype=torch.float32),
            actions,
            timesteps.to(dtype=torch.long),
            prompt=prompt
        )
            
        actions[-1] = action
        action = action.detach().cpu().numpy()

        #env.render()
        state, reward, done, infos = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        #rewards[-1] = reward
        
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

        infos['episode_length'] = episode_length

    return episode_return, episode_length



