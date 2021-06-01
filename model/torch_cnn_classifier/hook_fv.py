import torch


class HookFV(object):
    def __init__(self, module):
        self.data = torch.Tensor()
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.data = input

