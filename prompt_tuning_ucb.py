from ast import parse
import gym
import numpy as np
import torch
#import wandb
import argparse
import random
import sys, os
import time
import itertools
from datetime import datetime
import imageio
from tqdm import trange
import matplotlib.pyplot as plt
import pickle

from prompt_dt.prompt_decision_transformer import GoalTransformer
from prompt_dt.prompt_utils import eval_episodes, get_prompt_batch
from prompt_dt.prompt_tuning_utils import online_collect_episode, prompt_to_torch, sample_prompt_from_history

# env name to the utils module
# the utils module contains functions for: data&env loader, prompt&sequence batch loader
import envs.mazerunner.utils as mazerunner_utils
import envs.kitchen_toy.utils as kitchen_toy_utils
import envs.kitchen.utils as kitchen_utils
import envs.crafter.utils as crafter_utils
CONFIG_DICT = {
    'mazerunner': mazerunner_utils,
    'kitchen_toy': kitchen_toy_utils,
    'crafter': crafter_utils,
    'kitchen': kitchen_utils,
}


'''
UCB: use stochastic policies conditioned on different prompts to explore 
1. maintain the episode with best return to sample prompt from
2. maintain all prompts, choose the UCB prompt to collect next episode (stochastic)
3. keep the deterministic evaluation score of each prompt, return the best one as the optimized prompt.
'''
class UCB:
    def __init__(self, c=1.0):
        self.arms = []
        self.N = 0
        self.c = c

    def add_prompt(self, prompt, r=None, r_d=-1e8):
        if r is None:
            d = {'prompt': prompt, 'r': 0, 'n': 0, 'r_determ': r_d}
        else:
            d = {'prompt': prompt, 'r': r, 'n': 1, 'r_determ': r_d}
            self.N += 1 
        self.arms.append(d)

    def get_prompt(self):
        # intuition: more exploration with the task-goal prompt
        #if np.random.rand() < 0.5:
        #    return 0, self.arms[0]['prompt']

        # UCB
        R = [i['r']+self.c*np.sqrt(0.5*np.log(self.N+1e-8)/(i['n']+1e-8)) for i in self.arms]
        i = np.argmax(R)
        return i, self.arms[i]['prompt']

    def update(self, idx, r):
        last_r = self.arms[idx]['r']
        last_n = self.arms[idx]['n']
        self.arms[idx]['r'] = (last_r * last_n + r) / (last_n+1)
        self.arms[idx]['n'] += 1
        self.N += 1

    # get the arm with best deterministic evaluation return
    def get_arm_best_determ_return(self):
        R = [i['r_determ'] for i in self.arms]
        i = np.argmax(R)
        return self.arms[i]

    def print(self):
        print(['i:{} r:{} n:{} r_d:{}'.format(i, self.arms[i]['r'], self.arms[i]['n'], self.arms[i]['r_determ']) 
            for i in range(len(self.arms))])


def ucb_prompt_search(env, model, variant, device):
    ep_returns, ep_best_returns, ep_best_prompts = [], [], [] # online returns; best arm performances
    ucb = UCB(c=variant['c'])

    task_goal = env.get_task_prompt()
    prompt = prompt_to_torch(task_goal, variant['max_prompt_len'], device)
    ret, _, _, _ = online_collect_episode(env, model, variant['max_ep_len'], device, prompt, variant['render'], deterministic=True) # just evaluate the initial prompt performance
    ucb.add_prompt(prompt, r_d=ret)
    best_episode_goals, max_ep_ret = None, -1e8 # maintain goals from the best episode ever seen

    for i in range(variant['max_test_episode'] // 2):
        #ucb.print()
        # explore new traj with UCB
        i_prompt, prompt = ucb.get_prompt()
        ret, ep_len, episode_visited_goals, _ = online_collect_episode(env, model, variant['max_ep_len'], device, prompt, variant['render'], deterministic=False)
        ucb.update(i_prompt, ret) # update the UCB arm
        if variant['task']!=-1:
            print('ret: {}, best ret: {}, best arm performance: {}'.format(ret, max_ep_ret, ucb.get_arm_best_determ_return()['r_determ']))
            #env.print_episode_stats(prompt)
        if ret >= max_ep_ret:
            max_ep_ret = ret
            best_episode_goals = episode_visited_goals
        ep_returns.append(ret)
        
        # sample next prompt from best episode, evaluate it (deterministic), add to UCB arms
        prompt = sample_prompt_from_history(best_episode_goals, task_goal, variant['max_prompt_len'], device=device)
        ret, ep_len, episode_visited_goals, _ = online_collect_episode(env, model, variant['max_ep_len'], device, prompt, variant['render'], deterministic=True)
        best_arm = ucb.get_arm_best_determ_return()
        best_arm_ret, best_arm_prompt = best_arm['r_determ'], best_arm['prompt']
        if ret > best_arm_ret:
            ucb.add_prompt(prompt, ret, r_d=ret)
        if ret >= max_ep_ret:
            max_ep_ret = ret
            best_episode_goals = episode_visited_goals
        ep_returns.append(ret)
        ep_best_returns += [best_arm_ret]*2
        ep_best_prompts += [best_arm_prompt]*2
        #print(ep_best_returns)
    
    #best_prompt = ucb.get_arm_best_determ_return()['prompt']
    #print(ret, ep_len, episode_visited_goals)
    #imageio.mimsave(args.save_gif, rgbs, 'GIF', duration=0.03)
    return ep_best_prompts, ep_returns, ep_best_returns


def test(variant):
    device = variant['device']
    
    _, _, _, test_info, test_env_list, test_trajectories_list, _ = \
        CONFIG_DICT[args.env].get_train_test_dataset_envs(\
            args.dataset_path, device, max_ep_len = variant['max_ep_len'], only_test=False, 
            render_mode='human' if variant['render'] else 'rgb_array')

    K = variant['K']
    max_ep_len = variant['max_ep_len']
    assert K==max_ep_len, "currently, training context K should be == max episode length"
    print('Max ep length {}, training context length {}'.format(variant['max_ep_len'], K))

    state_dim = test_info[0]['state_dim'] #test_env_list[0].observation_space.shape[0]
    act_dim = test_info[0]['act_dim'] #test_env_list[0].action_space.shape[0]
    action_space = test_env_list[0].action_space
    prompt_dim = test_env_list[0].prompt_dim
    print('state {} action {} prompt goal {}'.format(state_dim, act_dim, prompt_dim))

    model = GoalTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        action_space=action_space,
        prompt_dim=prompt_dim,
        max_length=K,
        max_ep_len=variant['max_ep_len'], 
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    model = model.to(device=device)

    model.load_state_dict(torch.load(args.load_path, map_location=device), strict=True)
    print('model initialized from: ', args.load_path)
    #print(model)

    # showcase on a task
    if args.task!=-1:
        env = test_env_list[args.task]
        best_prompt, ep_returns, ep_best_returns = ucb_prompt_search(env, model, variant, device)
    
        x = np.arange(0, len(ep_returns))
        plt.xlabel('test episode')
        plt.ylabel('return')
        plt.plot(x, ep_returns, label='return', c='b')
        plt.plot(x, ep_best_returns, label='best return', c='r')
        if hasattr(CONFIG_DICT[args.env], 'get_oracle_returns'):
            oracle_return = np.mean(CONFIG_DICT[args.env].get_oracle_returns([test_trajectories_list[args.task]]))
            plt.plot(x, [oracle_return]*len(x), label='oracle', c='g')
        plt.legend()
        plt.savefig('{}_return.png'.format(args.env))

    # evaluate on all test tasks
    else:
        save_dir = os.path.join(args.save_dir, args.env)
        if args.env=='mazerunner':
            save_dir += str(test_env_list[0].maze_dim)
        os.makedirs(save_dir, exist_ok=True)

        data_save = []
        ep_returns, ep_best_returns = np.zeros(args.max_test_episode), np.zeros(args.max_test_episode)
        
        if args.n_tasks!=-1:
            test_env_list = test_env_list[0:args.n_tasks]
        for i in range(len(test_env_list)):
            env = test_env_list[i]
            best_prompts, rets, best_rets = ucb_prompt_search(env, model, variant, device)
            ep_returns += np.asarray(rets)
            ep_best_returns += np.asarray(best_rets)
            data_save.append({'ep_returns': ep_returns, 'prompts': best_prompts, 'prompts_performance': best_rets})
            print('task {} initial_prompt_ret {} optimized_prompt_ret {}'.format(i, best_rets[0], best_rets[-1]))

        x = np.arange(0, len(ep_returns))
        plt.xlabel('test episode')
        plt.ylabel('return')
        plt.plot(x, ep_best_returns/len(test_env_list), label='best return', c='r')
        if hasattr(CONFIG_DICT[args.env], 'get_oracle_returns'):
           oracle_return = np.mean(CONFIG_DICT[args.env].get_oracle_returns(test_trajectories_list))
           plt.plot(x, [oracle_return]*len(x), label='oracle', c='g')
        plt.legend()
        plt.savefig('{}/{}_s{}.png'.format(save_dir, args.fig_prefix, args.seed))
        with open('{}/{}_s{}.pkl'.format(save_dir, args.fig_prefix, args.seed), 'wb') as f:
            pickle.dump(data_save, f)
            

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='mazerunner') 
    parser.add_argument('--dataset_path', type=str, default='envs/mazerunner/mazerunner-d15-g4-4-t64-multigoal-astar.pkl')
    parser.add_argument('--fig_prefix', type=str, default='ucb') # save fig prefix 
    parser.add_argument('--seed', type=int, default=1)
    #parser.add_argument('--test_optimal_prompt', action='store_true', default=False) # use 'optimal_prompts' saved in trajectories for test?
    parser.add_argument('--task', type=int, default=-1) # test task index, -1 means all tasks
    parser.add_argument('--n_tasks', type=int, default=-1) # number of test tasks, -1 means all tasks
    #parser.add_argument('--prompt_len', type=int, default=1) # prompt length
    parser.add_argument('--max_test_episode', type=int, default=100) # number of adaptation episodes
    parser.add_argument('--save_dir', type=str, default='eval_results') 
    parser.add_argument('--c', type=float, default=1) # UCB c

    parser.add_argument('--render', action='store_true', default=False) 
    parser.add_argument('--save-gif', type=str, default='test.gif') 
    parser.add_argument('--load-path', type=str, 
        default='model_saved/gym-experiment-mazerunner-mazerunner-d15-g4-4-t64-multigoal-astar-20240116113910/prompt_model_mazerunner_iter_4999')

    parser.add_argument('--max_prompt_len', type=int, default=5) # max len of sampled prompt
    parser.add_argument('--max_ep_len', type=int, default=64) # max episode len in both dataset & env
    parser.add_argument('--K', type=int, default=64) # max Transformer context len (the whole sequence is max_prompt_len+K)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    test(variant=vars(args))
