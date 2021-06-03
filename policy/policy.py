from policy import dqn
from policy import ppo
#from policy import sac
from tianshou.policy import MultiAgentPolicyManager


# a function to get policy for each configuration
def get_policy(cfg, input_shape, action_space, name = 'CartPole'):
    if cfg.algo == 'dqn':
        return dqn.get_dqn_policy(cfg, input_shape, action_space, name)

    if cfg.algo == 'ppo':
        return ppo.get_ppo_policy(cfg, input_shape, action_space, name)

    #if cfg.algo == 'sac':
   #     return sac.get_sac_policy(cfg, input_shape, action_space, name)



class ExplorationRateDecayPolicy():
    '''
        Callable class for exploration decay during training (to matches API in trainer)
    '''
    def __init__(self, policy, max_epoch, step_per_epoch, mile_stones = (0.05,0.5), rates = (0.2,0.1,0.05)):
        self.policy = policy
        self.total = max_epoch + step_per_epoch
        self.mile__stones = mile_stones
        self.rates = rates

    def set_eps(self, r):
        if isinstance(self.policy, MultiAgentPolicyManager):
            for single_policy in self.policy.policies:
                if hasattr(single_policy, 'set_eps'):
                    single_policy.set_eps(r)
        else:
            if hasattr(self.policy, 'set_eps'):
                self.policy.set_eps(r)

    def __call__(self, epoch, steps):
        ratio = steps/self.total
        for index,k in enumerate(self.mile__stones):
            if ratio < k:
                self.set_eps(self.rates[index])
                break
        else:
            self.set_eps(self.rates[-1])

