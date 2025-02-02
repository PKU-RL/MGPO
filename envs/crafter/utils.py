import os
import pickle
import torch
import numpy as np
from crafter.env import Env


achievements = ['collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling',
                'collect_stone', 'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow',
                'eat_plant', 'make_iron_pickaxe', 'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
                'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant', 'place_stone',
                'place_table', 'wake_up']


class CrafterEvalEnv(Env):
    def __init__(self, seed, final_goal=None):
        super().__init__(seed=seed)
        self.env_name = 'crafter'
        self.episode_visited_goals, self.episode_visited_goal_ids = [], []
        self.achievement_completion = [0 for _ in range(len(achievements))]

        # self.final_goal = final_goal

    def reset(self):
        self._episode = 0
        obs = super().reset()
        obs = obs.transpose(2, 0, 1).flatten() / 255.0
        self.achievement_completion = [0 for _ in range(len(achievements))]
        self.episode_visited_goals, self.episode_visited_goal_ids = [], []
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = obs.transpose(2, 0, 1).flatten() / 255.0
        for i in range(len(info['achievements'])):
            if info["achievements"][achievements[i]] > 0 and self.achievement_completion[i] == 0:
                subgoal = np.zeros(len(achievements), dtype=np.float32)
                self.achievement_completion[i] = 1
                subgoal[i] = 1
                self.episode_visited_goals.append(np.asarray(subgoal))
                self.episode_visited_goal_ids.append(i)

        return obs, reward, done, info

    def get_task_prompt(self):
        return [np.ones(len(achievements), dtype=np.float32)]
        # return self.final_goal


def get_env_list(trajectories_list, device, max_ep_len):
    infos, env_list = [], []
    for trajectory in trajectories_list:
        env = CrafterEvalEnv(
            seed=trajectory['info']['seed'], final_goal=trajectory['info']['prompt'][-1])
        if not hasattr(env, 'prompt_dim'):
            setattr(env, 'prompt_dim', len(achievements))
        env_list.append(env)
        info = {
            'max_ep_len': max_ep_len,
            'state_dim': np.prod(env.observation_space.shape),
            'act_dim': env.action_space.n,
            'device': device,
            'prompt_dim': len(achievements),
            'discrete_action': True,
        }
        infos.append(info)
    return infos, env_list


def get_train_test_dataset_envs(dataset_dir, device, max_ep_len=500, num_test_env=50, **kwargs):
    data_list = os.listdir(dataset_dir)

    trajectories_list = []
    for data_file in data_list:
        with open(os.path.join(dataset_dir, data_file), 'rb') as f:
            trajectory = pickle.load(f)
            if trajectory['timesteps'] > max_ep_len:
                continue
            trajectory['observations'] = trajectory['observations'].reshape(
                trajectory['timesteps'], -1)
            trajectory['next_observations'] = trajectory['next_observations'].reshape(
                trajectory['timesteps'], -1)

            # 1-24 Update prompt, one-hot and multi-hot final goal
            trajectory['info']['prompt'].append(
                np.ones(len(achievements), dtype=np.float32))
            for i in range(len(trajectory['info']['prompt']) - 2, 0, -1):
                trajectory['info']['prompt'][i] -= trajectory['info']['prompt'][i-1]

            trajectories_list.append(trajectory)

    test_trajectories_list = trajectories_list[-num_test_env:] # unseen tasks to eval
    trajectories_list = trajectories_list[:-num_test_env] # offline data for training
    val_trajectories_list = trajectories_list[:num_test_env] # seen tasks to eval    

    val_info, val_env_list = get_env_list(
        val_trajectories_list, device, max_ep_len)
    test_info, test_env_list = get_env_list(
        test_trajectories_list, device, max_ep_len)

    return val_info, val_env_list, val_trajectories_list, test_info, test_env_list, test_trajectories_list, trajectories_list


def get_prompt(trajectory, max_prompt_length=23, prompt_length=None, device=None, use_optimal_prompt=False):
    if prompt_length is None:
        # sample a prompt length between [1,max_prompt_length] for training
        prompt_length = np.random.randint(1, max_prompt_length+1)

    goal_timesteps = []
    if prompt_length > len(trajectory['info']['prompt']):
        prompt_length = len(trajectory['info']['prompt'])
    if prompt_length > 1:
        goal_range = np.arange(0, len(trajectory['info']['prompt']) - 1)
        goal_timesteps = np.random.choice(
            goal_range, prompt_length - 1, replace=False).tolist()
        goal_timesteps.sort()
    goal_timesteps.append(len(trajectory['info']['prompt']) - 1)

    mask = np.concatenate([np.ones((1, prompt_length)), np.zeros(
        (1, max_prompt_length - prompt_length))], axis=1)

    prompt = []
    for t in goal_timesteps:
        prompt.append(trajectory['info']['prompt'][t])
    prompt = np.array(prompt).reshape(1, -1, len(achievements))
    prompt = np.concatenate(
        [np.zeros((1, max_prompt_length - prompt_length, len(achievements))), prompt], axis=1)
    prompt = torch.from_numpy(prompt).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).float().to(device=device)

    return prompt, mask


def get_oracle_returns(trajectories):
    return [len(trajectory['info']['prompt']) for trajectory in trajectories]


if __name__ == '__main__':
    get_train_test_dataset_envs('../../crafter_dataset', 'cpu')