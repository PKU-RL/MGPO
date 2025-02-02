'''
collect a dataset of maze trajectories
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from env import KitchenToyEnv, expert_policy, enum_tasks
from copy import deepcopy
import pickle
import random

def main(args, task_list, mode='train'):
    
    dataset = []
    ep_lens = []
    n_saved_ep = 0

    for i, task in enumerate(task_list):
        ep_data = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[], 'terminals':[], 'prompts_per_timestep': []}
        env = KitchenToyEnv(task=task)
        obs = env.reset()
        
        done=False
        step_cnt=0
        last_goal_t = 0 # last goal finished timestep
        while not done:
            action = expert_policy(env)
            step_cnt+=1
            next_obs, rew, terminated, info = env.step(action)
            if step_cnt>=args.max_ep_len:
                terminated=True
            ep_data['observations'].append(obs)
            ep_data['next_observations'].append(next_obs)
            ep_data['actions'].append(action)
            ep_data['rewards'].append(rew)
            ep_data['terminals'].append(terminated)
            obs = next_obs

            if info['success']: # a new goal is visited
                goal = env.episode_visited_goals[-1]
                for j in range(last_goal_t, step_cnt):
                    ep_data['prompts_per_timestep'].append(goal)
                last_goal_t = step_cnt

            if terminated:
                done=True
                break
        
        ep_lens.append(step_cnt)
        n_saved_ep+=1
        for k in ep_data:
            ep_data[k] = np.asarray(ep_data[k])
        ep_data['task'] = task
        ep_data['timesteps'] = step_cnt
            
        prompts = env.get_optimal_prompt()
        #print(prompts, task)
        ep_data['prompts'] = prompts

        #print(ep_data['prompts_per_timestep'], ep_data['observations'], ep_data['timesteps'], len(ep_data['prompts_per_timestep']))
            
        dataset.append(ep_data)
        #print(ep_data)

    if args.save_path:
        path = "{}_{}.pkl".format(args.save_path, mode)
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        print('Save episodes {}, transitions {}, max_ep_len {}'.format(n_saved_ep, np.sum(ep_lens), np.max(ep_lens)))
    plt.cla()
    plt.hist(x=ep_lens)
    plt.savefig('kitchen_toy_hist_max{}.png'.format(args.max_ep_len))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--n-episode', type=int, default=10000)
    parser.add_argument('--max-ep-len', type=int, default=90)
    parser.add_argument('--n-test-task', type=int, default=10) # all tasks 13650
    parser.add_argument('--save-path', type=str, default='kitchen_toy_t90')

    args = parser.parse_args()
    all_tasks = enum_tasks()
    random.seed(0)
    random.shuffle(all_tasks)
    #print(all_tasks[0:args.n_test_task])
    main(args, all_tasks[0:args.n_test_task], 'test')
    main(args, all_tasks[args.n_test_task:], 'train') 