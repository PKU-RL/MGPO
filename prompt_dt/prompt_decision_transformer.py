# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from .trajectory_gpt2 import GPT2Model

'''
goal-prompt transformer: 
input: goal sequence + (s,a,s,a,...)
predict: at each s, predict next a
'''
class GoalTransformer(nn.Module):

    def __init__(
            self,
            state_dim,
            act_dim,
            action_space,
            prompt_dim, # dimension for each goal in prompt
            hidden_size,
            max_length=None, # context len
            max_ep_len=None, # max timesteps for positional embedding
            #action_tanh=True,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.action_space = action_space
        self.discrete_action = True if hasattr(action_space, 'n') else False
        #print(self.discrete_action)
        self.prompt_dim = prompt_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        #self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        
        self.embed_prompt = torch.nn.Linear(self.prompt_dim, hidden_size)
        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        #self.prompt_embed_return = torch.nn.Linear(1, hidden_size)
        #self.prompt_embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        #self.prompt_embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if not self.discrete_action else []))
        )
        # updated 1-26: predict logstd for Gaussian
        #if not self.discrete_action:
        #    self.predict_action_logstd = nn.Linear(hidden_size, self.act_dim)
        #print(self.predict_action)
        #self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, timesteps, attention_mask=None, prompt=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        # convert action labels to onehot
        if self.discrete_action:
            #print(actions)
            actions = F.one_hot(actions, num_classes=self.act_dim).to(dtype=torch.float32)
            #print(actions.shape)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        #returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        #returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (s_1, a_1, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # the prompt is a sequence of goals: (g1,g2,...)
        if prompt is not None:
            prompt_goals, prompt_attention_mask = prompt
            prompt_seq_length = prompt_goals.shape[1]
            #prompt_state_embeddings = self.prompt_embed_state(prompt_states)
            #prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
            #if prompt_returns_to_go.shape[1] % 10 == 1:
            #    prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:,:-1])
            #else:
            #    prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_goal_embeddings = self.embed_prompt(prompt_goals) # (batch, prompt_len, dim)
            #prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)

            #prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            #prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
            #prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings
            #prompt_goal_embeddings = prompt_goal_embeddings + prompt_time_embeddings 

            #prompt_stacked_inputs = torch.stack(
            #    (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
            #).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            #prompt_stacked_attention_mask = torch.stack(
            #    (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
            #).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)

            # stacked_inputs add prompted sequence
            '''
            if prompt_stacked_inputs.shape[1] == 3 * seq_length: # if only smaple one prompt
                prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
                stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            else: # if sample one prompt for each traj in batch
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
            '''
            stacked_inputs = torch.cat((prompt_goal_embeddings, stacked_inputs), dim=1) # (batch, prompt_len+seq_len*2, dim)
            stacked_attention_mask = torch.cat((prompt_attention_mask, stacked_attention_mask), dim=1) # (batch, prompt_len+seq_len*2)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state'] # (batch, prompt_len+seq_len*2, dim)

        if prompt is None:
            # reshape x so that the second dimension corresponds to the original
            # states (0), actions (1); i.e. x[:,0,t] is the token for s_t
            x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        else:
            # remove the part for prompt sequence
            x = x[:, -seq_length*2:, :].reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        #return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        state_preds = self.predict_state(x[:,1])[:, -seq_length:, :]    # predict next state given state and action
        action_preds = self.predict_action(x[:,0])[:, -seq_length:, :]  # predict next action given state

        return state_preds, action_preds
        #if self.discrete_action:
        #    return state_preds, action_preds
        #else:
        #    action_logstd_preds = self.predict_action_logstd(x[:,0])[:, -seq_length:, :] 
        #    return state_preds, (action_preds, action_logstd_preds)

    def get_action(self, states, actions, timesteps, prompt=None, deterministic=True):
        states = states.reshape(1, -1, self.state_dim)
        if self.discrete_action:
            actions = actions.reshape(1, -1)
        else:
            actions = actions.reshape(1, -1, self.act_dim)
        timesteps = timesteps.reshape(1, -1)
        attention_mask = torch.ones(states.shape[1]).to(dtype=torch.long, device=states.device).reshape(1, -1)
    
        #print(states.shape, actions.shape, timesteps.shape, attention_mask, prompt)
        _, action_preds = self.forward(states, actions, timesteps, attention_mask, prompt)
        #print(action_preds)

        if self.discrete_action:
            if deterministic:
                ret = torch.argmax(action_preds[0,-1], dim=-1)
            else:
                action_dist = torch.distributions.categorical.Categorical(logits=action_preds[0,-1])
                ret = action_dist.sample()
        else:
            #action_preds, action_logstd_preds = action_preds
            if deterministic:
                ret = action_preds[0,-1]
            else:
                mean = action_preds[0,-1]
                std = torch.exp(-torch.ones_like(mean)) #torch.exp(action_logstd_preds[0,-1])
                action_dist = torch.distributions.Normal(mean, std)
                ret = action_dist.sample()
                #print(mean, std, ret)
        return ret


    '''Rollout in the env'''
    def on_env_reset(self, state, device):
        self.eval()
        self.to(device=device)
        self.env_states = torch.from_numpy(state).reshape(1, self.state_dim).to(device=device, dtype=torch.float32)
        if self.discrete_action:
            self.env_actions = torch.zeros((0,), device=device, dtype=torch.long)
        else:
            self.env_actions = torch.zeros((0, self.act_dim), device=device, dtype=torch.float32)
        #rewards = torch.zeros(0, device=device, dtype=torch.float32)
        self.env_timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    def on_env_get_action(self, prompt, device, deterministic=True):
        if self.discrete_action:
            self.env_actions = torch.cat([self.env_actions, torch.zeros((1,), device=device, dtype=torch.long)], dim=0)
        else:
            self.env_actions = torch.cat([self.env_actions, torch.zeros((1, self.act_dim), device=device)], dim=0)
        with torch.no_grad():
            max_traj_len = (1024-prompt[0].shape[1])//2
            if self.env_states.shape[0] > max_traj_len:
                print('warning: the trajectory is truncated to fit the max context length of GPT2')
                self.env_states = self.env_states[-max_traj_len:]
                self.env_actions = self.env_actions[-max_traj_len:]
                self.env_timesteps = self.env_timesteps[:,-max_traj_len:]
                #print(self.env_states.shape, self.env_actions.shape, self.env_timesteps.shape, prompt[0].shape)
            action = self.get_action(
                self.env_states.to(dtype=torch.float32),
                self.env_actions,
                self.env_timesteps.to(dtype=torch.long),
                prompt=prompt,
                deterministic=deterministic,
            )
        self.env_actions[-1] = action
        action = action.cpu().numpy()
        return action

    def on_env_step(self, state, reward, done, timestep, device):
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, self.state_dim)
        self.env_states = torch.cat([self.env_states, cur_state], dim=0)
        #rewards[-1] = reward
        self.env_timesteps = torch.cat([self.env_timesteps, 
            torch.ones((1, 1), device=device, dtype=torch.long) * (timestep+1)], dim=1)