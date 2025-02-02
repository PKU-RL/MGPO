import argparse
import os
import sys
import random
import yaml
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from crafter.env import Env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from achievement_distillation.algorithm import *
from achievement_distillation.model import *
from achievement_distillation.wrapper import VecPyTorch

from utils import achievements

def main(args):
    # Load config file
    config_file = open(args.config, "r")
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Fix Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # CUDA setting
    torch.set_num_threads(1)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    seeds = list(range(args.n_episode))
    # env = Env(seed=seeds[0])
    env = Env(seed=args.seed)
    venv = DummyVecEnv([lambda: env])
    venv = VecPyTorch(venv, device=device)

    # Create model
    model_cls = getattr(sys.modules[__name__], config["model_cls"])
    model: BaseModel = model_cls(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        **config["model_kwargs"],
    )
    model.to(device)
    # print(model)

    # Load checkpoint
    ckpt_path = args.model_path
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    # Eval
    model.eval()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for ep in tqdm(range(args.n_episode)):
        env = Env(seed=seeds[ep])
        venv = DummyVecEnv([lambda: env])
        venv = VecPyTorch(venv, device=device)
        
        obs = venv.reset()
        states = torch.zeros(1, config["model_kwargs"]["hidsize"]).to(device)
        # with open('./world.txt', 'a') as f:
        #     for i in range(64):
        #         for j in range(64):
        #             f.write(f'{str(venv.envs[0]._world[i, j][0])}, {type(venv.envs[0]._world[i, j][1])}')
        #             f.write('\n')
        step_cnt = 0
        ep_data = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'info': {}
        }

        completion = [0 for _ in range(len(achievements))]
        prompt = []

        while True:
            outputs = model.act(obs, states=states)
            latents = outputs["latents"]
            actions = outputs["actions"]
            next_obs, rewards, dones, infos = venv.step(actions)
            
            # example:
            # obs.shape = torch.Size([1, 3, 64, 64]) actions.shape = torch.Size([1, 1]) rewards.shape = torch.Size([1, 1]) dones.shape = torch.Size([1, 1])
            # info
            # [
            #   {
            #       'inventory':
            #            {'health': 0, 'food': 6, 'drink': 4, 'energy': 3, 'sapling': 0,'wood': 0, 'stone': 0,'coal': 3,'iron': 0,'diamond': 0,'wood_pickaxe': 1,'stone_pickaxe': 1,'iron_pickaxe': 0,'wood_sword': 1,'stone_sword': 1,'iron_sword': 0},
            #       'achievements':
            #            {
            #             'collect_coal': 3, 'collect_diamond': 0, 'collect_drink': 3, 'collect_iron': 0, 'collect_sapling': 1,
            #             'collect_stone': 19, 'collect_wood': 10, 'defeat_skeleton': 1, 'defeat_zombie': 2, 'eat_cow': 2,
            #             'eat_plant': 0, 'make_iron_pickaxe': 0, 'make_iron_sword': 0, 'make_stone_pickaxe': 1, 'make_stone_sword': 1,
            #             'make_wood_pickaxe': 1, 'make_wood_sword': 1, 'place_furnace': 0, 'place_plant': 1, 'place_stone': 17,
            #             'place_table': 3, 'wake_up': 0
            #             }
            #       'discount': 0.0,
            #       'semantic': array(), dtype=uint8),
            #       'player_pos': array([29, 44]),
            #       'reward': -0.1,
            #       'terminal_observation': array(), dtype=uint8)
            #   }
            # ]
            
            info, = infos
            # Update prompt
            for i in range(len(info['achievements'])):
                if info["achievements"][achievements[i]] > 0 and completion[i] == 0:
                    completion[i] = 1
                    prompt.append(np.array(completion, dtype=np.float32))
                    break
            
            ep_data["observations"].append(obs.squeeze().cpu().numpy())
            ep_data["actions"].append(actions.squeeze().item())
            ep_data["rewards"].append(rewards.squeeze().item())
            ep_data["terminals"].append(1 if dones.any() else 0)
            ep_data['next_observations'].append(next_obs.squeeze().cpu().numpy())
            step_cnt += 1

            # Done
            if dones.any():
                break

            # Update states
            if (rewards > 0.1).any():
                with torch.no_grad():
                    next_latents = model.encode(next_obs)
                states = next_latents - latents
                states = F.normalize(states, dim=-1)


            obs = next_obs

        ep_data['observations'] = np.array(ep_data['observations'])
        ep_data['next_observations'] = np.array(ep_data['next_observations'])
        ep_data['actions'] = np.array(ep_data['actions'])
        ep_data['rewards'] = np.array(ep_data['rewards'])
        ep_data['terminals'] = np.array(ep_data['terminals'])

        ep_data['timesteps'] = step_cnt
        ep_data['info']['seed'] = seeds[ep]
        ep_data['info']['prompt'] = prompt

        with open(os.path.join(args.save_dir, f'{ep}.pkl'), 'wb') as f:
            pickle.dump(ep_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episode", type=int, default=2000)
    parser.add_argument("--config", type=str, default="./configs/ppo_ad.yaml")
    parser.add_argument("--model_path", type=str,
                        default='./models/agent-e250.pt')
    parser.add_argument("--save_dir", type=str, default="./crafter_dataset/")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    main(args)
