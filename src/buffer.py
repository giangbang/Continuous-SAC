import numpy as np
import torch
from collections import namedtuple


Transition = namedtuple(
    "Transition", ("states", "actions", "rewards", "next_states", "dones")
)


# https://github.com/denisyarats/pytorch_sac_ae/blob/master/utils.py
class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device="auto"):
        self.capacity = capacity
        self.batch_size = batch_size
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            device = self.device
        else:
            self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=bool)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        """Add a new transition to replay buffer"""
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], terminated)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        """Sample batch of Transitions with batch_size elements.
        Return a named tuple with 'states', 'actions', 'rewards', 'next_states' and 'dones'.
        """
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        dones = torch.as_tensor(self.dones[idxs], device=self.device).float()

        return Transition(obses, actions, rewards, next_obses, dones)

    def get_sample_without_replace_iter(self):
        """
        Sample without replacement a batch of Transitions with batch_size elements.
        Return a named tuple with 'states', 'actions', 'rewards', 'next_states' and 'dones'.
        """
        while True:
            current_len = self.capacity if self.full else self.idx
            _indx = np.arange(current_len)
            np.random.shuffle(_indx)
            for batch_start_indx in range(0, current_len, self.batch_size):
                idxs = _indx[
                    batch_start_indx : min(
                        current_len, batch_start_indx + self.batch_size
                    )
                ]

                obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
                actions = torch.as_tensor(self.actions[idxs], device=self.device)
                rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
                next_obses = torch.as_tensor(
                    self.next_obses[idxs], device=self.device
                ).float()
                dones = torch.as_tensor(self.dones[idxs], device=self.device)

                yield Transition(obses, actions, rewards, next_obses, dones)

    def sample_without_replace(self):
        if getattr(self, "_sample_wo_replace", None) is None:
            self._sample_wo_replace = self.get_sample_without_replace_iter()
        return next(self._sample_wo_replace)
