# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn.functional as F
import time
from wandb import env
#from .prompt_utils import flatten_prompt
import copy

class PromptSequenceTrainer:

    def __init__(self, model, optimizer, batch_size,
                 scheduler=None, eval_fns=None, get_prompt=None, get_prompt_batch=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        #self.get_batch = get_batch
        if model.discrete_action:
            self.loss_fn = F.cross_entropy
        else:
            self.loss_fn = lambda a_hat, a: torch.mean((a_hat - a) ** 2)
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.get_prompt = get_prompt
        #self.prompt = self.get_prompt() # sample prompt data when initialization
        self.get_prompt_batch = get_prompt_batch

        self.start_time = time.time()


    def train(self, num_steps, no_prompt=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step(no_prompt)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs


    def train_step(self, no_prompt=False):
        prompt, batch = self.get_prompt_batch()
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch
        action_target = torch.clone(actions)
        if no_prompt:
            raise NotImplementedError
            #state_preds, action_preds, reward_preds = self.model.forward(
            #    states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=None
            #)
        else:
            state_preds, action_preds = self.model.forward(
                states, actions, timesteps, attention_mask=attention_mask, prompt=prompt
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        if self.model.discrete_action:
            action_target = action_target.reshape(-1)[attention_mask.reshape(-1) > 0]
        else:
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        #print(action_preds.shape, action_target.shape, attention_mask.shape)
        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        #with torch.no_grad():
        #    self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
        self.diagnostics['training/action_error'] = loss.detach().cpu().item()

        return loss.detach().cpu().item()


    def eval_iteration_multienv(self, trajectories_list, eval_episodes, info, variant, 
                                env_list, iter_num=0, print_logs=False, no_prompt=False, group='test'):
        #print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating at {} tasks: {}'.format(len(env_list), group))
        self.model.eval()

        eval_start = time.time()
        eval_prompt_lens = [i for i in range(1, variant['max_prompt_len']+1)]
        # evaluate for different prompt length, average over envs
        for prompt_len in eval_prompt_lens:
            #self.prompt = self.get_prompt(trajectories_list[env_id], variant['max_prompt_len'], 
            #                            prompt_length=1, device=info[env_id]['device'])
            outputs = eval_episodes(info[0], variant, env_list, self.model, prompt_len, 
                                    self.get_prompt, trajectories_list)
            for k, v in outputs.items():
                logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

 
    def save_model(self, env_name, postfix, folder):
        model_name = '/prompt_model_' + env_name + postfix
        torch.save(self.model.state_dict(),folder+model_name)  # model save
        print('model saved to ', folder+model_name)
