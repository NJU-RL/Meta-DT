import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from meta_sac.model import ContextGaussianPolicy, ContextQNetwork



class SAC_Offline(object):
    def __init__(self, env, context_dim, hidden_dim, lr, alpha, gamma, tau, device='cpu'):

        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu') 

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.critic = ContextQNetwork(state_dim, action_dim, context_dim, hidden_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = ContextQNetwork(state_dim, action_dim, context_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = ContextGaussianPolicy(state_dim, action_dim, context_dim, hidden_dim, action_space=env.action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=0.1*lr)


    def update_parameters(self, states, actions, rewards, next_states, masks, contexts, next_contexts):

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states, next_contexts)
            qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_action, next_contexts)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + masks * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(states, actions, contexts)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(states, contexts)

        qf1_pi, qf2_pi = self.critic(states, pi, contexts)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()

