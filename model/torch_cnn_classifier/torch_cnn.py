import torch
import torch.nn as nn

from model.torch_cnn_classifier.get_torch_model import get_torch_model
from model.torch_cnn_classifier.hook_fv import HookFV


class TorchCNN(torch.nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False, **kwargs):
        super().__init__()
        self.name = model_name
        self.num_classes = num_classes
        model, head, aux, input_dim, fv_dim = get_torch_model(model_name, num_classes, pretrained)
        self.model = model
        self.head = head
        self.aux = aux
        self.input_dim = input_dim
        self.fv_dim = fv_dim

        if type(self.head) == nn.ModuleList:
            self.features_hook = [HookFV(layer) for layer in self.head]
        else:
            self.features_hook = HookFV(self.head)

    def forward(self, x):
        return self.model(x)

    def forward_head(self, x, aux=True):
        if type(self.head) == nn.ModuleList:
            if aux:
                output = []
                for i in range(len(self.head)):
                    output.append(self.head[i](x[i]))
                return output
            if type(x) == list:
                return self.head[0](x[0])
            else:
                return self.head[0](x)
        else:
            return self.head(x)

    def feature_vectors(self, x, aux=True):
        _ = self.model(x)
        if type(self.head) == nn.ModuleList:
            if aux:
                return [hook.data[0] for hook in self.features_hook]
            else:
                return self.features_hook[0].data[0]
        else:
            return self.features_hook.data[0]

    def train(self, mode=True):
        self.model.train(mode)

    def freeze_model(self, mode=True):
        for param in self.model.parameters():
            param.requires_grad = not mode

    def freeze_head(self, mode=True):
        if type(self.head) == nn.ModuleList:
            for layer in self.head:
                for param in layer.parameters():
                    param.requires_grad = not mode
        else:
            for param in self.head.parameters():
                param.requires_grad = not mode

    def freeze_backbone(self, mode=True):
        self.freeze_model(mode)
        self.freeze_head(False)

    def freeze_aux(self, mode=True):
        if self.aux is not None:
            for module in self.aux:
                for param in module.parameters():
                    param.requires_grad = not mode

    def parameters(self):
        return self.model.parameters()

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict)

    def state_dict(self):
        return self.model.state_dict()
