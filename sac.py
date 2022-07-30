import torch
import torch.nn as nn
from model import Actor, Critic
import torch.nn.functional as F
import numpy as np


# https://github.com/denisyarats/pytorch_sac_ae/blob/master/sac_ae.py
class SAC:
    def __init__(self, 
        obs_shape: np.ndarray,
        action_shape: np.ndarray,
        device='cpu',
        hidden_dim=50,
        discount=0.99,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        num_layers=3
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        
        self.actor  = Actor(obs_shape, action_shape, num_layers, 
                hidden_dim, actor_log_std_min, actor_log_std_max).to(device)
                
        self.critic = Critic(obs_shape, action_shape, num_layers, hidden_dim).to(device)
        
        self.target_entropy = -np.prod(action_shape)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )
        
        self.log_ent_coef = torch.log(torch.ones(1, device=device)).requires_grad_(True)
        
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], 
            lr=alpha_lr, betas=(alpha_beta, 0.999)
        )
        
        self.train()
        
        
    def train(self):
        '''
        Set training mode for actor and critic
        The behavior of actor will be changed between train and eval modes
        Note: this function does not cause any update in the weights of actor
        and critic
        '''
        self.actor.train()
        self.critic.train()
        
    def eval(self):
        self.actor.eval()
        self.critic.eval()
        
    def _update_critic(self, batch):
        # Compute target Q 
        with torch.no_grad():
            next_pi, next_log_pi  = self.actor.sample(batch.next_states, compute_log_pi=True)
            next_q_vals = self.critic.target_q(batch.next_states, next_pi)
            next_q_val  = torch.minimum(*next_q_vals)
            
            ent_coef    = torch.exp(self.log_ent_coef)
            next_q_val  = next_q_val - ent_coef * next_log_pi
            
            target_q_val= batch.rewards + (1-batch.dones)*self.discount*next_q_val
            
        current_q_vals  = self.critic.online_q(batch.states, batch.actions)
        critic_loss     = .5*sum(F.mse_loss(current_q, target_q_val) for current_q in current_q_vals)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.critic.polyak_update(self.critic_tau)
        
        return critic_loss.item()
        
    def _update_actor(self, batch):
        pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)
        
        q_vals = self.critic.online_q(batch.states, pi)
        q_val  = torch.minimum(*q_vals)
        
        with torch.no_grad():
            ent_coef = torch.exp(self.log_ent_coef)
        
        actor_loss = (ent_coef * log_pi - q_val).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, batch):
        with torch.no_grad():
            pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)
        alpha_loss = -(self.log_ent_coef * (log_pi + self.target_entropy).detach()).mean()
        
        self.ent_coef_optimizer.zero_grad()
        alpha_loss.backward()
        self.ent_coef_optimizer.step()
        
        return alpha_loss.item()
        
    def update(self, buffer):
        batch = buffer.sample()
        
        critic_loss = self._update_critic(batch)
        actor_loss = self._update_actor(batch)
        alpha_loss = self._update_alpha(batch)
        
        return critic_loss, actor_loss, alpha_loss
        
     
    def select_action(self, state):
        with torch.no_grad():
            return self.actor.sample(state)[0].cpu().numpy()