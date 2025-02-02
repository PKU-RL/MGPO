'''
An A*-search policy for partially-observed maze

A grid has 4 possible types: visited, unvisited, wall, unknown.
once the agent reaches an unvisited grid, the unknown area is reduced
at each step, the policy selects the best unvisited grid to reach:
max G+H, G: shortest path from current to the unvisited grid, H: dx+dy distance between unvisited and the goal.
'''

import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
import random

class AStarSolver:
    def __init__(self, env):
        self.maze_dim = env.maze_dim
        self.reset()

    def reset(self):
        # 0: unknown 1: visited 2: unvisited 3: wall
        self.state = np.zeros((self.maze_dim, self.maze_dim), dtype=int)
        self.unvisited = {} # unvisited_pos: A*_score

    def observe(self, pos, maze):
        i, j = tuple(pos)
        self.state[i,j] = 1 # visited
        self.unvisited.pop((i,j), None) # remove from unvisited list

        space_west = 0
        seek_west = j - 1
        while seek_west >= 0:
            if maze[i, seek_west] == 0:
                if self.state[i, seek_west]==0:
                    self.state[i, seek_west]=2 # unvisited
                    self.unvisited[(i,seek_west)] = -1
                seek_west -= 1
                space_west += 1
            else:
                self.state[i, seek_west]=3 # wall
                break

        space_east = 0
        seek_east = j + 1
        while seek_east < self.maze_dim:
            if maze[i, seek_east] == 0:
                if self.state[i, seek_east]==0:
                    self.state[i, seek_east]=2 # unvisited
                    self.unvisited[(i,seek_east)] = -1
                seek_east += 1
                space_east += 1
            else:
                self.state[i, seek_east]=3 # wall
                break

        space_north = 0
        seek_north = i - 1
        while seek_north >= 0:
            if maze[seek_north, j] == 0:
                if self.state[seek_north, j]==0:
                    self.state[seek_north, j]=2 # unvisited
                    self.unvisited[(seek_north, j)] = -1
                seek_north -= 1
                space_north += 1
            else:
                self.state[seek_north, j]=3 # wall
                break

        space_south = 0
        seek_south = i + 1
        while seek_south < self.maze_dim:
            if maze[seek_south, j] == 0:
                if self.state[seek_south, j]==0:
                    self.state[seek_south, j]=2 # unvisited
                    self.unvisited[(seek_south, j)] = -1
                seek_south += 1
                space_south += 1
            else:
                self.state[seek_south, j]=3 # wall
                break

    def render(self, env, pause=0.1):
        if not env._plotting:
            env.start_plotting()
            plt.ion()
            env._plotting = True

        plt.tight_layout()
        background = np.ones((self.maze_dim, self.maze_dim, 3), dtype=np.uint8)
        maze_img = (
            background * abs(np.expand_dims(env.maze, -1) - 1) * 255
        )  # zero out (white) where there is a valid path

        state_colors = [
            None, 
            [200,240,240], # visited
            [252,230,202], # unvisited
            [156,102,31], # wall
        ]
        for i in range(self.maze_dim):
            for j in range(self.maze_dim):
                s = self.state[i,j]
                if s!=0:
                    maze_img[i,j,:] = state_colors[s]

        # TODO: give us more colors
        goal_color_wheel = [
            [240, 3, 252],
            [255, 210, 87],
            [3, 219, 252],
            [252, 2157, 3],
        ]
        for i, goal_pos in enumerate(env.goal_positions):
            if env.active_goal_idx > i:
                continue
            x, y = goal_pos
            maze_img[x, y, :] = goal_color_wheel[i % len(goal_color_wheel)]

        maze_img[env.pos[0], env.pos[1], :] = [110, 110, 110]  # grey
        plt.imshow(maze_img)

        for i, goal_pos in enumerate(env.goal_positions):
            if env.active_goal_idx > i:
                continue
            y, x = goal_pos
            env._goal_render_texts[i] = plt.text(
                x, y, str(i), ha="center", va="center"
            )

        #env._ax.set_title(
        #    f"k={env.goal_positions}, active_goal={env.goal_positions[env.active_goal_idx]}"
        #)
        plt.draw()
        plt.pause(pause)

    # Dijkstra shortest paths from start pos to all visited/unvisited pos
    def find_shortest_paths(self, pos):
        ret = [[{'vis': False, 'pa': None, 'act': None, 'dis': 1e8} for i in range(self.maze_dim)] for j in range(self.maze_dim)]
        px, py = tuple(pos)
        motions = [[0, -1], [-1, 0], [0, 1], [1, 0]]
        q = Queue()
        q.put((px,py,0))
        ret[px][py]['vis']=True
        ret[px][py]['dis']=0
        while not q.empty():
            x,y,d = q.get()
            for i, m in enumerate(motions):
                x_, y_ = x+m[0], y+m[1]
                if not (x_>=0 and x_<self.maze_dim and y_>=0 and y_<self.maze_dim):
                    continue
                if (not ret[x_][y_]['vis']) and (
                    self.state[x_,y_]==1 or self.state[x_,y_]==2):
                    ret[x_][y_]['vis'] = True 
                    ret[x_][y_]['pa'] = (x,y)
                    ret[x_][y_]['act'] = i
                    ret[x_][y_]['dis'] = d+1
                    q.put((x_,y_,d+1))
        '''
        for i in ret:
            for j in i:
                if j['vis']:
                    print(j)
        '''
        return ret

    # given Dijkstra results, end_pos, return action sequence
    def plan(self, shortest_paths, pos_e):
        ret = []
        x,y = tuple(pos_e)
        while shortest_paths[x][y]['pa'] is not None:
            ret.append(shortest_paths[x][y]['act'])
            x,y = shortest_paths[x][y]['pa']
        ret.reverse()
        return ret

    # search for an unvisited grid, plan for actions
    def search(self, pos, goal_pos=None):
        px, py = tuple(pos)
        # if the task goal_pos is given: use A* to pick an optimal unvisited grid towards the goal
        if goal_pos is not None:
            gx, gy = tuple(goal_pos)
            shortest_paths = self.find_shortest_paths(pos)

            if self.state[gx, gy]==1 or self.state[gx, gy]==2:
                actions = self.plan(shortest_paths, goal_pos)
                return gx, gy, actions

            for k in self.unvisited:
                x, y = k
                G = shortest_paths[x][y]['dis']
                H = abs(x-gx)+abs(y-gy) # estimate lower bound of dis(subgoal, goal)
                self.unvisited[k] = G+H
            ans = min(self.unvisited, key=lambda x: self.unvisited[x])
            actions = self.plan(shortest_paths, ans)
            return ans[0], ans[1], actions

        # if goal_pos is None: randomly pick an unvisited grid to explore the maze 
        else:
            #unvisited = list(self.unvisited.keys())
            #ans = random.choice(unvisited)
            shortest_paths = self.find_shortest_paths(pos)
            for k in self.unvisited:
                x, y = k
                self.unvisited[k] = shortest_paths[x][y]['dis']
            #ans = min(self.unvisited, key=lambda x: self.unvisited[x])
            ans_sorted = list(self.unvisited.keys())
            ans_sorted.sort(key=lambda x: self.unvisited[x])
            if len(ans_sorted)>5:
                ans_sorted = ans_sorted[:5]
            ans = random.choice(ans_sorted) # randomly pick from the 5 nearest unvisited pos
            #print(ans_sorted, ans)
            actions = self.plan(shortest_paths, ans)
            return ans[0], ans[1], actions



if __name__=='__main__':
    from mazerunner import MazeRunnerEnv
    env = MazeRunnerEnv(maze_dim=15, min_num_goals=1, max_num_goals=3)
    astar = AStarSolver(env)

    env.reset()
    astar.observe(env.pos, env.maze)

    Done=False
    cnt=0
    while not Done:
        _, _, actions = astar.search(env.pos, env.goal_positions[env.active_goal_idx])
        #print(astar.state)
        for a in actions:
            astar.render(env)
            plt.savefig('astar/{}.png'.format(cnt))
            cnt+=1
            input()
            obs, rew, terminated, truncated, info = env.step(a)
            astar.observe(env.pos, env.maze)
            if terminated:
                Done=True
                break
    astar.render(env)
    plt.savefig('astar/{}.png'.format(cnt))