import torch
import torch.nn as nn
from model import Actor, Critic
import torch.nn.functional as F
import numpy as np


# https://github.com/denisyarats/pytorch_sac_ae/blob/master/sac_ae.py
class SAC:
    def __init__(
        self,
        obs_shape: np.ndarray,
        action_shape: np.ndarray,
        device="auto",
        hidden_dim=50,
        discount=0.99,
        alpha_lr=3e-4,
        actor_lr=3e-4,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        critic_lr=3e-4,
        critic_tau=0.005,
        num_layers=3,
        init_temperature=1,
        reward_scale=1.0,
        *args,
        **kwargs
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            device = self.device
        else:
            self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.reward_scale = reward_scale

        use_rbn = getattr(self, "use_batchrenorm", False)

        self.actor = Actor(
            obs_shape,
            action_shape,
            num_layers,
            hidden_dim,
            actor_log_std_min,
            actor_log_std_max,
            use_batchrenorm=False,
        ).to(device)

        self.critic = Critic(
            obs_shape,
            action_shape,
            num_layers,
            hidden_dim,
            use_batchrenorm=use_rbn,
        ).to(device)

        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic._online_q.parameters(),
            lr=critic_lr,
        )

        self.log_ent_coef = torch.log(
            init_temperature * torch.ones(1, device=device)
        ).requires_grad_(True)

        self.ent_coef_optimizer = torch.optim.Adam(
            [self.log_ent_coef],
            lr=alpha_lr,
        )

        self.train()

    def train(self):
        """
        Set training mode for actor and critic
        The behavior of actor will be changed between train and eval modes
        Note: this function does not cause any update in the weights of actor
        and critic
        """
        self.actor.train()

    def eval(self):
        self.actor.eval()

    def _update_critic(self, batch):
        self.critic.train(True)
        # Compute target Q
        with torch.no_grad():
            next_pi, next_log_pi = self.actor.sample(
                batch.next_states, compute_log_pi=True
            )
            next_q_vals = self.critic.target_q(batch.next_states, next_pi)
            next_q_val = torch.minimum(*next_q_vals)

            ent_coef = torch.exp(self.log_ent_coef)
            next_q_val = next_q_val - ent_coef * next_log_pi

            target_q_val = (
                self.reward_scale * batch.rewards
                + (1 - batch.dones) * self.discount * next_q_val
            )

        current_q_vals = self.critic.online_q(batch.states, batch.actions)
        critic_loss = 0.5 * sum(
            F.mse_loss(current_q, target_q_val) for current_q in current_q_vals
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.polyak_update(self.critic_tau)

        self.critic.train(False)

        return critic_loss.item()

    def _update_actor(self, batch):
        self.actor.train(True)
        pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)

        q_vals = self.critic.online_q(batch.states, pi)
        q_val = torch.minimum(*q_vals)

        with torch.no_grad():
            ent_coef = torch.exp(self.log_ent_coef)

        actor_loss = (ent_coef * log_pi - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.train(False)

        return actor_loss.item()

    def _update_alpha(self, batch):
        with torch.no_grad():
            pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)
            log_pi = log_pi.mean()
        alpha_loss = -(self.log_ent_coef * (log_pi + self.target_entropy).detach())
        self.entropy = -log_pi

        self.ent_coef_optimizer.zero_grad()
        alpha_loss.backward()
        self.ent_coef_optimizer.step()

        return alpha_loss.item()

    def update(self, get_batch):
        actor_losses, critic_losses, alpha_losses = [], [], []

        batch = get_batch()

        critic_loss = self._update_critic(batch)
        actor_loss = self._update_actor(batch)
        alpha_loss = self._update_alpha(batch)

        critic_losses.append(critic_loss)
        actor_losses.append(actor_loss)
        alpha_losses.append(alpha_loss)

        return np.mean(critic_losses), np.mean(actor_losses), np.mean(alpha_losses)

    def select_action(self, state, deterministic=False):
        self.actor.train(False)
        state = state.to(self.device)
        with torch.no_grad():
            return (
                self.actor.sample(state, deterministic=deterministic)[0].cpu().numpy()
            )
