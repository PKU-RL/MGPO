''' 
utils for prompt-tuning during online adaptation 
'''

import numpy as np
#import gym
import json, pickle, random, os, torch
from collections import namedtuple
#from .prompt_evaluate_episodes import prompt_evaluate_episode, prompt_evaluate_episode_rtg
import random
from copy import deepcopy
import torch
from .prompt_utils import get_sequence


'''
collect an episode given a prompt
return: 
    reward, length, subgoals visited in this episode, episode data: Dict
'''
def online_collect_episode(
        env,
        model,
        max_ep_len=1000,
        device='cuda',
        prompt=None,
        render=True,
        deterministic=True,
    ):
    if env.env_name=='mazerunner':
        prompt_np_pos = (prompt[0].cpu().numpy()[0]*env.maze_dim).astype(int)
        mask_np = prompt[1].cpu().numpy()[0].astype(int)
        si = (mask_np!=0).argmax()
        render_fn = lambda : env.render_with_prompt(prompt_np_pos[si:], return_rgb=True)
    elif env.env_name=='kitchen_toy':
        #print('task:', env.task)
        #print('prompt:', prompt)
        render_fn = lambda : env.render(return_rgb=True)
    else:
        render_fn = lambda : env.render(return_rgb=True)

    ep_data = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[], 'terminals':[]}
    rgbs = []
    state = env.reset()
    model.on_env_reset(state, device)
    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        action = model.on_env_get_action(prompt, device, deterministic=deterministic)
        if render and t%3==0:
            rgbs.append(render_fn())
        next_state, reward, done, infos = env.step(action)
        episode_length += 1
        if episode_length>=max_ep_len:
            done = True
        model.on_env_step(next_state, reward, done, t, device)

        ep_data['observations'].append(state)
        ep_data['next_observations'].append(next_state)
        ep_data['actions'].append(action)
        ep_data['rewards'].append(reward)
        ep_data['terminals'].append(done)
        episode_return += reward
        state = next_state
        if done:
            break
    if render:
        rgbs.append(render_fn())
    for k in ep_data:
        ep_data[k] = np.asarray(ep_data[k])
    ep_data['timesteps'] = episode_length
    ep_data['rgbs'] = rgbs
    return episode_return, episode_length, env.episode_visited_goals, ep_data


# convert a prompt of List[numpy array] into padded (prompt, mask) to feed into the Transformer
def prompt_to_torch(prompt, max_prompt_length=5, device=None):
    prompt_length = len(prompt)
    prompt_dim = prompt[0].shape[0]
    assert prompt_length <= max_prompt_length

    mask = np.concatenate([np.zeros((1, max_prompt_length - prompt_length)), np.ones((1, prompt_length))], axis=1)
    prompt = np.array(prompt).reshape(1,-1,prompt_dim)
    prompt = np.concatenate([np.zeros((1, max_prompt_length - prompt_length, prompt_dim)), prompt], axis=1)

    prompt = torch.from_numpy(prompt).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(device=device)
    #print(prompt, mask)
    return prompt, mask

# sample prompt from a history goal list: [sampled subgoals] + [task goal]; convert to torch (prompt, mask)
def sample_prompt_from_history(prompt_list, task_goal, max_prompt_length=5, prompt_length=None, device=None):
    #print(prompt_list, task_goal)
    if prompt_length is None:
        # sample a prompt length between [1,max_prompt_length] for training
        prompt_length = np.random.randint(1,max_prompt_length+1)
    prompt_dim = task_goal[0].shape[0]

    goal_timesteps = []
    if prompt_length > len(prompt_list)+1: # if prompt_length exceeds goals in history
        prompt_length = len(prompt_list)+1
    if prompt_length>1:
        goal_range = np.arange(0, len(prompt_list))
        goal_timesteps = np.random.choice(goal_range, prompt_length-1, replace=False).tolist()
        goal_timesteps.sort()
    
    # padding to the left
    mask = np.concatenate([np.zeros((1, max_prompt_length - prompt_length)), np.ones((1, prompt_length))], axis=1)
    prompt = []
    for t in goal_timesteps:
        prompt.append(prompt_list[t])
    prompt += task_goal
    prompt = np.array(prompt).reshape(1,-1,prompt_dim)
    prompt = np.concatenate([np.zeros((1, max_prompt_length - prompt_length, prompt_dim)), prompt], axis=1)

    prompt = torch.from_numpy(prompt).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(device=device)
    #print(prompt, goal_timesteps, mask)
    return prompt, mask

# sample multiple prompts from a history goal list. return a batch of (prompt, mask)
def sample_n_prompts_from_history(n, prompt_list, task_goal, max_prompt_length=5, prompt_length=None, device=None):
    p_list, p_mask_list = [], []
    for i in range(n):
        p, p_mask = sample_prompt_from_history(prompt_list, task_goal, max_prompt_length, prompt_length, device)
        p_list.append(p)
        p_mask_list.append(p_mask)
    prompt, mask = torch.cat(p_list, dim=0), torch.cat(p_mask_list, dim=0)
    return prompt, mask


# compute logp of joint actions in a trajectory given different prompts
# input a batch of prompt, compute logp(actions|trajectory, p) for all prompt. return log_probs (batchsize,)
def get_trajectory_action_logp(trajectory, prompt, model, info, variant):
    bsz = prompt[0].shape[0]
    states, actions, rewards, dones, rtg, timesteps, attention_mask = get_sequence(trajectory, variant['K'], info)
    #print(actions.expand(bsz,-1), timesteps.expand(bsz,-1), attention_mask.expand(bsz,-1))
    with torch.no_grad():
        state_preds, action_preds = model.forward(
            states.expand(bsz,-1,-1), 
            actions.expand(bsz,-1) if info['discrete_action'] else actions.expand(bsz,-1,-1), 
            timesteps.expand(bsz,-1), 
            attention_mask=attention_mask.expand(bsz,-1), 
            prompt=prompt
        )
    
    action_preds = action_preds[:, :trajectory['timesteps']]
    if info['discrete_action']:
        actions = actions.expand(bsz,-1)[:, :trajectory['timesteps']]
        action_dist = torch.distributions.categorical.Categorical(logits=action_preds)
        logp = action_dist.log_prob(actions)
        #print(torch.exp(torch.sum(logp)))
        #print(torch.argmax(action_preds, dim=-1), actions)
    else:
        actions = actions.expand(bsz,-1,-1)[:, :trajectory['timesteps']]
        raise NotImplementedError

    return torch.sum(logp, dim=-1).cpu().numpy()