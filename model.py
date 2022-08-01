import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    '''
    Multi-layer perceptron
    '''
    def __init__(self, inputs_dim, outputs_dim, n_layer, n_unit):
        super().__init__()
        self.inputs_dim     = inputs_dim
        self.outputs_dim    = outputs_dim
        
        net = [nn.Linear(inputs_dim, n_unit), nn.ReLU()]
        for _ in range(n_layer-2):
            net.append(nn.Linear(n_unit, n_unit))
            net.append(nn.ReLU())
        net.append(nn.Linear(n_unit, outputs_dim))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    '''
    Actor class for state 1d inputs
    '''
    def __init__(self, inputs_dim, output_dims, n_layer, n_unit,
                log_std_min, log_std_max):
        super().__init__()
        self.inputs_dim     = inputs_dim
        self.output_dims    = output_dims
        self.log_std_min    = log_std_min
        self.log_std_max    = log_std_max
        
        self._actor = MLP(inputs_dim, output_dims*2, n_layer, n_unit)
        
    def forward(self, x):
        return self._actor(x).chunk(2, dim=-1)
        
    def sample(self, x, compute_log_pi=False):
        mu, log_std = self.forward(x)
        
        if not self.training: return torch.tanh(mu), None
        
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        
        Gaussian_distribution = torch.distributions.normal.Normal(
                                mu, log_std.exp())
                                
        sampled_action  = Gaussian_distribution.rsample()
        squashed_action = torch.tanh(sampled_action)
        
        if not compute_log_pi: return squashed_action, None
        
        log_pi_normal   = Gaussian_distribution.log_prob(sampled_action)
        log_pi_normal   = torch.sum(log_pi_normal, dim=-1, keepdim=True)
        
        # See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
        log_squash      = log_pi_normal - torch.sum(
                            torch.log(
                                F.relu(1 - squashed_action ** 2) + 1e-6
                            ),
                            dim = -1, keepdim=True
                        )
        assert len(log_squash.shape) == 2 and len(squashed_action.shape) == 2
        assert log_squash.shape == log_pi_normal.shape
        return squashed_action, log_squash

class DoubleQNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_layer, n_unit, requires_grad=True):
        super().__init__()
        inputs_dim = state_dim + action_dim
        
        self.q1 = MLP(inputs_dim, 1, n_layer, n_unit)
        self.q2 = MLP(inputs_dim, 1, n_layer, n_unit)
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, x, a):
        assert x.shape[0] == a.shape[0]
        assert len(x.shape) == 2 and len(a.shape) == 2
        x = torch.cat([x, a], dim=1)
        return self.q1(x), self.q2(x)
    
        
class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, n_layer, n_unit):
        super().__init__()
        
        self._online_q = DoubleQNet(obs_shape, action_shape, n_layer, n_unit)
        self._target_q = DoubleQNet(obs_shape, action_shape, n_layer, n_unit, requires_grad=False)
        
        self._target_q.load_state_dict(self._online_q.state_dict())
        
    def target_q(self, x, a): return self._target_q(x, a)
    
    def online_q(self, x, a): return self._online_q(x, a)
    
    def polyak_update(self, tau):
        for target, online in zip(self._target_q.parameters(), self._online_q.parameters()):
            target.data.copy_(target.data * (1-tau) + tau * online.data)
    