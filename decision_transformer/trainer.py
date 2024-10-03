import numpy as np
import torch
from tqdm import tqdm
import time
import torch.nn.functional as F



class DT_Trainer(object):

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask):
        self.model.train()
        
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, 
            actions, 
            rewards, 
            rtg[:,:-1], 
            timesteps, 
            attention_mask=attention_mask,
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        loss = F.mse_loss(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()