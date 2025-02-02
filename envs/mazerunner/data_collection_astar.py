'''
collect a dataset of maze trajectories
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from mazerunner import MazeRunnerEnv
from maze_astar_solver import AStarSolver
from copy import deepcopy
import pickle

def main(args):
    env = MazeRunnerEnv(maze_dim=args.maze_dim, min_num_goals=1, max_num_goals=args.max_num_goals)
    astar = AStarSolver(env)

    dataset = []
    ep_lens = []
    n_saved_ep = 0

    while n_saved_ep<args.n_episode:
        ep_data = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[], 'terminals':[]}

        obs, info = env.reset()
        astar.reset()
        astar.observe(env.pos, env.maze)
        maze, goal_pos = deepcopy(env.maze), deepcopy(env.goal_positions)

        done=False
        step_cnt=0
        while not done:
            goal = None if args.random_explore else env.goal_positions[env.active_goal_idx]
            _, _, actions = astar.search(env.pos, goal)
            for a in actions:
                #astar.render(env, pause=0.01)
                step_cnt+=1
                next_obs, rew, terminated, truncated, info = env.step(a)
                ep_data['observations'].append(obs)
                ep_data['next_observations'].append(next_obs)
                ep_data['actions'].append(a)
                ep_data['rewards'].append(rew)
                ep_data['terminals'].append(terminated)
                obs = next_obs
                astar.observe(env.pos, env.maze)
                if terminated:
                    done=True
                    break
        #astar.render(env)
        
        # ignore too short or long episodes
        if step_cnt>=args.min_ep_len and step_cnt<=args.max_ep_len:
            ep_lens.append(step_cnt)
            n_saved_ep+=1
            for k in ep_data:
                ep_data[k] = np.asarray(ep_data[k])
            ep_data['goal_pos'] = goal_pos
            ep_data['maze'] = maze
            ep_data['timesteps'] = step_cnt
            dataset.append(ep_data)

    if args.save_path:
        with open(args.save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('Save episodes {}, transitions {}'.format(n_saved_ep, np.sum(ep_lens)))
    plt.hist(x=ep_lens)
    plt.savefig('maze_hist_min{}_max{}.png'.format(args.min_ep_len, args.max_ep_len))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episode', type=int, default=10000)
    parser.add_argument('--min-ep-len', type=int, default=20)
    parser.add_argument('--max-ep-len', type=int, default=50)
    parser.add_argument('--save-path', type=str, default='mazerunner-d15-g1-astar.pkl')
    parser.add_argument('--maze-dim', type=int, default=15)
    parser.add_argument('--max-num-goals', type=int, default=1)
    parser.add_argument('--random-explore', action="store_true")

    args = parser.parse_args()
    main(args)

    