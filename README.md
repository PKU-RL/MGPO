# MGPO: Pre-Trained Multi-Goal Transformers with Prompt Optimization for Efficient Online Adaptation

Official implementation of **[NeurIPS 2024]** [Pre-Trained Multi-Goal Transformers with Prompt Optimization for Efficient Online Adaptation](https://openreview.net/forum?id=DHucngOEe3).

![](figs/mgpo.png)

Previous works in skill pre-training utilize offline, task-agnostic dataset to accelerate RL. However, these approaches still require substantial RL steps to learn a new task. We propose MGPO, a method that leverages the power of Transformer-based policies to model sequences of goals during offline pre-training, enabling efficient online adaptation through prompt optimization.

## Installation
- Create a conda environment with `python==3.8.5`. Install python packages in `requirements.txt`.
- For the Crafter environment, `pip install crafter==1.8.3` or see the [repo](https://github.com/danijar/crafter).
- Download datasets from [link](xxxxx).


## Offline Pre-Training
Train the Transformer policy on MazeRunner-15 as an example: run `python train.py --env mazerunner --dataset_path 'dataset/mazerunner-d15-g4-4-t64-multigoal-astar.pkl' --max_prompt_len 5 --K 64 --max_ep_len 64 --batch_size 64 --test_optimal_prompt --subsample_trajectory --subsample_min_len 10`

Training scripts for all environments are listed in `train.sh`.

## Online Adaptation with Prompt Optimization
Test the policy on MazeRunner-15 using MGPO-UCB as an example: run `python prompt_tuning_ucb.py --env mazerunner --dataset_path 'dataset/mazerunner-d15-g4-4-t64-multigoal-astar.pkl' --max_prompt_len 5 --K 64 --max_ep_len 64 --load-path 'model_saved/gym-experiment-xxx/prompt_model_mazerunner_iter_4999' --task -1 --max_test_episode 100`, where `--load-path` is the model checkpoint dir.

Test scripts for all environments with MGPO-Random, MGPO-UCB, and MGPO-BPE are listed in `test.sh`.


## Using Your Own Environment & Dataset
Make an environment directory `envs/$ENV_NAME` and place all files related to the env wrapper and dataset here. To train the model in this env, you should implement the file `envs/$ENV_NAME/utils.py` with two functions: 1. `get_train_test_dataset_envs()`, 2. `get_prompt()`. A step-by-step guidance is as follows.

#### Env wrapper
It should be a gym-style env. See `envs.mazerunner.utils.MazeRunnerEvalEnv` as an example.
- `action_space`, `observation_space` should be gym Box or Discrete space
- `discrete_action`: bool variable, whether the action space is discrete
- `prompt_dim`: dimension of each goal in the prompt
- `reset()`: return observation
- `step(action)`: return obs, reward, done, info

#### Dataset format
Run the data collection code for MazeRunner and see the saved pkl for example.
The pickle dataset is a List [traj 1, ..., traj n], each element is a trajectory. Each trajectory is a `Dict` with following keys:
- `timesteps`: an integer T of number of transitions in this trajectory.
- `observations`, `actions`. `next_observations`, `rewards`, `terminals`: numpy array of shape (T, obs_dim); (T, act_dim) for continuous action or (T,) for discrete action; (T, obs_dim); (T,); (T,) respectively. Containing the transitions in this trajectory.
- `optimal_prompts`: (optional) if you want to test the online performance with some better prompts, save the prompts in this key. For example, in MazeRunner, this is a numpy array (K, 2) of the goal paths(x,y positions) of a shortest path from start to the final pos.
- Necessary information to reproduce the env collecting this trajectory. E.g. seed. In MazeRunner, these keys are `maze` and `goal_pos`. Because for test, we will make test envs according to some trajectories in the dataset, and the prompt will be sampled from these trajectories, respectively.

In my implementation of MazeRunner and MetaWorld in prompt-dt, the total number of transitions in dataset is about ~300K.

#### `get_train_test_dataset_envs()` in utils.py
See the mazerunner implementation as an example.
`def get_train_test_dataset_envs(dataset_path, device, max_ep_len, n_train_env=10, n_test_env=10)`
- The function returns `info, env_list, val_trajectories_list, test_info, test_env_list, test_trajectories_list, trajectories_list`.
- It loads the dataset pickle file. Split the dataset into n_test_env test trajectories (`test_trajectories_list`) and other trajectories for training (`trajectories_list`).
- In `trajectories_list`, we select another n_train_env trajs (`val_trajectories_list`) as seen tasks during training. 
- We reproduce envs for `val_trajectories_list` and `test_trajectories_list` respectively, using the env task information saved in each episode. The resulting env list are`env_list` and `test_env_list`, each is a list of env wrappers for online test.
- `info` and `test_info` are list of `Dict` for each env and test_env. Each element is the same dict containing keys: `max_ep_len`, `state_dim`, `act_dim`, `device`, `prompt_dim`, `discrete_action`

#### `get_prompt()` in utils.py
See the mazerunner implementation as an example.
`def get_prompt(trajectory, max_prompt_length, prompt_length=None, device=None, use_optimal_prompt=False)`
- this function samples a prompt in a trajectory and process it into a sequence to feed into the Transformer.
- `trajectory`: an element in the dataset list
- `max_prompt_length`: the max number of goals in prompt (Updated 1-19: including the last token for the task goal). pad prompt to this length
- `prompt_length`: if None, sample a prompt_length between [1,max_prompt_length] (Updated 1-19: including the last token for the task goal)
- `device`: the torch device
- `use_optimal_prompt`: sample prompt from `trajectory['optimal_prompts']`? This is optional, you don't need to implement the True case if you do not set --test_optimal_prompt to run `train.py`.
- This function first samples a prompt (prompt_length, prompt_dim) according to the trajectory. Then pad the prompt sequence to length max_prompt_length on the left, the shape becomes (max_prompt_length, prompt_dim). The  prompt mask (max_prompt_length,) indicates whether each position is a valid goal.  E.g,  [0,0,1,1,1] means the sampled prompt has 3 goals, and the left two goals are padded goals.
- Return: `prompt` (max_prompt_length, prompt_dim), `mask` (max_prompt_length,). All torch tensors on the `device`.

#### Finally
In the beginning of `train.py`: 
- add `import envs.$ENV_NAME.utils as $ENV_NAME_utils`
- add to the `CONFIG_DICT`: `'$ENV_NAME': $ENV_NAME_utils`

Then your implemented functions will be automatically called when you set `--env` to your $ENV_NAME. You can run `train.py` to train the model. The training loss and test results can be visualized in wandb.


## Citation
If you find this code useful, please cite our paper:
```bibtex
@inproceedings{yuanpre,
  title={Pre-Trained Multi-Goal Transformers with Prompt Optimization for Efficient Online Adaptation},
  author={Yuan, Haoqi and Fu, Yuhui and Xie, Feiyang and Lu, Zongqing},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
Our implementation is based on [PromptDT](https://github.com/mxu34/prompt-dt). You may also cite this work.