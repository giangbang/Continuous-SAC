from .sac import SAC


class CrossQ(SAC):
    def __init__(self, *args, **kwargs):
        self.use_rebatchnorm = True
        super().__init__(*args, **kwargs)
        self.critic_tau = 1
        print("CrossQ: use Rebatch Normalization = True")
        print("CrossQ: use Target Network = False")

    def _update_critic(self, batch):
        pass
