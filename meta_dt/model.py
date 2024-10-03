import numpy as np
import torch
import torch.nn as nn
import transformers

from meta_dt.trajectory_gpt2 import GPT2Model


class MetaDecisionTransformer(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self, 
            state_dim, 
            act_dim, 
            hidden_size, 
            context_dim=16, 
            max_length=None, 
            max_ep_len=4096, 
            action_tanh=True, 
            **kwargs
        ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.max_length = max_length


        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.state_encoder = torch.nn.Linear(state_dim, context_dim)
        self.embed_state = torch.nn.Linear(context_dim*2, hidden_size)
        # self.embed_state = torch.nn.Linear(state_dim, hidden_size)
        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.prompt_embed_return = torch.nn.Linear(1, hidden_size)
        self.prompt_embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.prompt_embed_action = torch.nn.Linear(self.act_dim, hidden_size)


        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, contexts, actions, rewards, returns_to_go, timesteps, attention_mask=None,prompt=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        # state_embeddings = self.embed_state(states)1
        state_encoding = self.state_encoder(states)
        state_embeddings = self.embed_state(torch.cat((state_encoding, contexts), dim=-1))
        # state_embeddings = self.embed_state(states)

        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        #####
        # process prompt the same as d-t
        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards,  prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]
            prompt_state_embeddings = self.prompt_embed_state(prompt_states)
            prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
            if prompt_returns_to_go.shape[1] % 10 == 1:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:,:-1])
            else:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)
            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
            prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings

            prompt_stacked_inputs = torch.stack(
                (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            prompt_stacked_attention_mask = torch.stack(
                (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
            ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)

            # stacked_inputs add prompted sequence
            if prompt_stacked_inputs.shape[1] == 3 * seq_length: # if only smaple one prompt
                prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
                stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            else: # if sample one prompt for each traj in batch
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        if prompt is None:
            # reshape x so that the second dimension corresponds to the original
            # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]  # predict next action given state

        # get predictions
        # return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        # state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        # action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    # def get_action(self, states, contexts, actions, rewards, returns_to_go, timesteps, prompt,args,epoch):
    def get_action(self, states, contexts, actions, rewards, returns_to_go, timesteps, prompt,args,epoch,**kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        contexts = contexts.reshape(1, -1, self.context_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        prompt_length = args.prompt_length
        # prompt_states = states[:,-(prompt_length+self.max_length):-self.max_length]
        # prompt_actions = actions[:,-(prompt_length+self.max_length):-self.max_length]
        # prompt_rtg = returns_to_go[:,-(prompt_length+self.max_length):-self.max_length]
        # prompt_tsp = timesteps[:,-(prompt_length+self.max_length):-self.max_length]
        # prompt_mask =  torch.cat([torch.zeros(prompt_length-prompt_states.shape[1]), torch.ones(prompt_states.shape[1])])
        # prompt_mask = prompt_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
        # prompt_states = torch.cat(
        #         [torch.zeros((prompt_states.shape[0], prompt_length-prompt_states.shape[1], self.state_dim), device=prompt_states.device), prompt_states],
        #         dim=1).to(dtype=torch.float32)
        # prompt_actions = torch.cat(
        #         [torch.zeros((prompt_actions.shape[0], prompt_length - prompt_actions.shape[1], self.act_dim),
        #                      device=prompt_actions.device), prompt_actions],
        #         dim=1).to(dtype=torch.float32)
        # prompt_rtg = torch.cat(
        #         [torch.zeros((prompt_rtg.shape[0], prompt_length-prompt_rtg.shape[1], 1), device=prompt_rtg.device), prompt_rtg],
        #         dim=1).to(dtype=torch.float32)
        # prompt_tsp = torch.cat(
        #         [torch.zeros((prompt_tsp.shape[0], prompt_length-prompt_tsp.shape[1]), device=prompt_tsp.device), prompt_tsp],
        #         dim=1
        #     ).to(dtype=torch.long)
        
        if self.max_length is not None:
            states = states[:,-self.max_length:]
            contexts = contexts[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            contexts = torch.cat(
                [torch.zeros((contexts.shape[0], self.max_length-contexts.shape[1], self.context_dim), device=contexts.device), contexts],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        # prompt_mask =  attention_mask[:,-(prompt_length+self.max_length):-self.max_length]
        # prompt_states = states[:,-prompt_length:]
        # prompt_actions = actions[:,-prompt_length:]
        # prompt_rtg = returns_to_go[:,-prompt_length:]
        # prompt_tsp = timesteps[:,-prompt_length:]
        # prompt_mask =  attention_mask[:,-prompt_length:]
        # prompt = prompt_states,prompt_actions,None,prompt_rtg, prompt_tsp,prompt_mask

        # _, action_preds, return_preds = self.forward(
        #     states, contexts, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
        if epoch<=args.warm_train:
            _, action_preds, return_preds = self.forward(
                states,contexts,  actions, None, returns_to_go, timesteps, attention_mask=attention_mask,prompt=None)
        else:
            _, action_preds, return_preds = self.forward(
                states,contexts,  actions, None, returns_to_go, timesteps, attention_mask=attention_mask,prompt=prompt)

        return action_preds[0,-1]