# Code snippet source: : https://doi.org/10.5281/zenodo.8210150
# Original author: xinghd
# Copyright: For learning/research purposes only, following the original license.

"""
base_opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999)) # Any optimizer
lookahead = Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead
lookahead.zero_grad()
loss_function(model(input), target).backward() # Self-defined loss function
lookahead.step()
"""

from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        defaults = dict(alpha=alpha, k=k, counter=0)
        self.optimizer = optimizer
        self.defaults = optimizer.defaults
        self.defaults.update(defaults)
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        self._optimizer_step_pre_hooks = optimizer._optimizer_step_pre_hooks  # 添加这行_标注
        self._optimizer_step_post_hooks = optimizer._optimizer_step_post_hooks
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
            "_optimizer_step_pre_hooks": self._optimizer_step_pre_hooks,
            "_optimizer_step_post_hooks": self._optimizer_step_post_hooks
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
             "_optimizer_step_pre_hooks": state_dict["_optimizer_step_pre_hooks"],
            "_optimizer_step_post_hooks": state_dict["_optimizer_step_post_hooks"]
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)