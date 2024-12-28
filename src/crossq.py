from sac import SAC
import torch
import torch.nn.functional as F


class CrossQ(SAC):
    def __init__(self, *args, **kwargs):
        self.use_batchrenorm = True
        super().__init__(*args, **kwargs)
        print("CrossQ: use Rebatch Normalization = True")
        print("CrossQ: use Target Network = False")

    def _update_critic(self, batch):
        self.critic.train(True)

        with torch.no_grad():
            # next_action, next_log_pi, _ = self.actor.get_action(next_obs)
            next_action, next_log_pi = self.actor.sample(
                batch.next_states, compute_log_pi=True
            )
            all_obses = torch.concatenate([batch.states, batch.next_states], dim=0)
            all_actions = torch.concatenate([batch.actions, next_action], dim=0)

        all_Q1, all_Q2 = self.critic.online_q(all_obses, all_actions)
        Q1, target_Q1 = all_Q1.chunk(2, dim=0)
        Q2, target_Q2 = all_Q2.chunk(2, dim=0)

        with torch.no_grad():
            next_q_val = torch.minimum(target_Q1.detach(), target_Q2.detach())
            ent_coef = torch.exp(self.log_ent_coef)
            next_q_val = next_q_val - ent_coef * next_log_pi.detach()
            target_q_val = self.reward_scale * batch.rewards + (1 - batch.dones) * (
                self.discount * next_q_val
            )

        critic_loss = 0.5 * (
            F.mse_loss(Q1, target_q_val) + F.mse_loss(Q2, target_q_val)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.train(False)

        return critic_loss.item()
