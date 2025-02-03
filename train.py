from ast import parse
import gym
import numpy as np
import torch
import wandb
import argparse
import random
import sys, os
import time
import itertools
from datetime import datetime
from tqdm import trange

from prompt_dt.prompt_decision_transformer import GoalTransformer
from prompt_dt.prompt_seq_trainer import PromptSequenceTrainer
#from prompt_dt.prompt_utils import get_env_list
#from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune
#from prompt_dt.prompt_utils import process_total_data_mean, load_data_prompt, process_info
from prompt_dt.prompt_utils import eval_episodes, get_prompt_batch

# env name to the utils module
# the utils module contains functions for: data&env loader, prompt&sequence batch loader
import envs.mazerunner.utils as mazerunner_utils
import envs.kitchen_toy.utils as kitchen_toy_utils
import envs.crafter.utils as crafter_utils
import envs.kitchen.utils as kitchen_utils
CONFIG_DICT = {
    'mazerunner': mazerunner_utils,
    'kitchen_toy': kitchen_toy_utils,
    'crafter': crafter_utils,
    'kitchen': kitchen_utils,
}

def experiment(
        exp_prefix,
        variant,
):
    device = variant['device']
    log_to_wandb = variant['log_to_wandb']

    ######
    # construct train and test environments, datasets
    ######
    
    cur_dir = os.getcwd()
    #config_save_path = os.path.join(cur_dir, 'config')
    #data_save_path = os.path.join(cur_dir, 'data')
    save_path = os.path.join(cur_dir, 'model_saved/')
    if not os.path.exists(save_path): os.mkdir(save_path)
    
    info, env_list, val_trajectories_list, test_info, test_env_list, test_trajectories_list, trajectories_list = \
        CONFIG_DICT[args.env].get_train_test_dataset_envs(\
            args.dataset_path, device, max_ep_len = variant['max_ep_len'])

    print(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    print(f'Env List: {env_list} \n\n Test Env List: {test_env_list}')

    K = variant['K']
    assert K==variant['max_ep_len'], "currently, training context K should be == max episode length"
    batch_size = variant['batch_size']
    print('Max ep length {}, training context length {}, batch size {}'.format(variant['max_ep_len'], K, batch_size))


    ######
    # construct dt model and trainer
    ######
    exp_prefix = exp_prefix + '-' + args.env
    #num_env = len(train_env_name_list)
    #group_name = f'{exp_prefix}-{str(num_env)}-Env-{dataset_mode}'
    dataset_name = variant['dataset_path'].split('/')[-1].split('.')[0] # ds filename without .pkl
    time_now = datetime.now().strftime("%Y%m%d%H%M%S")
    group_name = f'{exp_prefix}-{dataset_name}' # wandb group name
    exp_prefix = f'{exp_prefix}-{dataset_name}-{time_now}' # wandb exp name

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

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    #env_name = train_env_name_list[0]
    trainer = PromptSequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        #get_batch=get_batch(trajectories_list[0], info[env_name], variant),
        scheduler=scheduler,
        #loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=None,
        get_prompt=CONFIG_DICT[args.env].get_prompt,
        get_prompt_batch=get_prompt_batch(trajectories_list, test_info[0], variant, CONFIG_DICT[args.env].get_prompt)
    )


    if not variant['evaluation']:
        ######
        # start training
        ######
        if log_to_wandb:
            wandb.init(
                name=exp_prefix,
                group=group_name,
                project='goal-dt',
                config=variant
            )
            save_path += wandb.run.name
            os.mkdir(save_path)

        # construct model post fix
        '''
        model_post_fix = '_TRAIN_'+variant['train_prompt_mode']+'_TEST_'+variant['test_prompt_mode']
        if variant['no_prompt']:
            model_post_fix += '_NO_PROMPT'
        '''
        model_post_fix = ''
        
        for iter in trange(variant['max_iters']):
            # train for many batches
            outputs = trainer.train(
                num_steps=variant['num_steps_per_iter'], 
                no_prompt=False #args.no_prompt
                )

            # start evaluation
            if iter % args.test_eval_interval == 0 and iter>0:
                # evaluate on unseen test tasks
                test_eval_logs = trainer.eval_iteration_multienv(test_trajectories_list,
                    eval_episodes, test_info, variant, test_env_list, iter_num=iter + 1, 
                    print_logs=True, no_prompt=False, group='test')
                outputs.update(test_eval_logs)

                # evaluate on some training tasks
                if args.test_on_training_tasks:
                    train_eval_logs = trainer.eval_iteration_multienv(val_trajectories_list,
                        eval_episodes, info, variant, env_list, iter_num=iter + 1, 
                        print_logs=True, no_prompt=False, group='train')
                    outputs.update(train_eval_logs)

            if iter % variant['save_interval'] == 0:
                trainer.save_model(
                    env_name=args.env, 
                    postfix=model_post_fix+'_iter_'+str(iter), 
                    folder=save_path)

            outputs.update({"global_step": iter}) # set global step as iteration

            if log_to_wandb:
                wandb.log(outputs)
        
        trainer.save_model(env_name=args.env,  postfix=model_post_fix+'_iter_'+str(iter),  folder=save_path)

    else:
        ####
        # start evaluating
        ####
        saved_model_path = os.path.join(save_path, variant['load_path'])
        model.load_state_dict(torch.load(saved_model_path), strict=True)
        print('model initialized from: ', saved_model_path)
        eval_iter_num = int(saved_model_path.split('_')[-1])

        eval_logs = trainer.eval_iteration_multienv(test_trajectories_list, eval_episodes, 
            test_info, variant, test_env_list, iter_num=eval_iter_num, print_logs=True, 
            no_prompt=False, group='test')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='gym-experiment')
    parser.add_argument('--env', type=str, default='mazerunner')
    parser.add_argument('--dataset_path', type=str, default='envs/mazerunner/mazerunner-d15-g1-t50-astar.pkl')
    parser.add_argument('--test_optimal_prompt', action='store_true', default=False) # use 'optimal_prompts' saved in trajectories for test?

    parser.add_argument('--evaluation', action='store_true', default=False) 
    parser.add_argument('--render', action='store_true', default=False) 
    parser.add_argument('--load-path', type=str, default= None) # choose a model when in evaluation mode

    parser.add_argument('--max_prompt_len', type=int, default=5) # max len of sampled prompt
    parser.add_argument('--max_ep_len', type=int, default=50) # max episode len in both dataset & env
    parser.add_argument('--K', type=int, default=50) # max Transformer context len (the whole sequence is max_prompt_len+K)
    parser.add_argument('--subsample_trajectory', action='store_true', default=False) # subsample during training?
    parser.add_argument('--subsample_min_len', type=int, default=-1) # subsample traj[0:l], l~U[min_len, traj_len]

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=5) 
    parser.add_argument('--max_iters', type=int, default=5000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--test_on_training_tasks', action='store_true', default=False) # eval on training tasks in addition to test tasks?
    parser.add_argument('--test_eval_interval', type=int, default=200)
    parser.add_argument('--save-interval', type=int, default=500)

    args = parser.parse_args()
    experiment(args.exp_name, variant=vars(args))
