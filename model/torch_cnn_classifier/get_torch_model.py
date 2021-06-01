import inspect
import types
import torchvision
import torch.nn as nn


def list_available_models():
    names = []
    for member in inspect.getmembers(torchvision.models):
        if type(member[1]) == types.FunctionType:
            names.append(member[0])
    return names


def get_torch_model(model_name, num_classes, pretrained=False):
    for member in inspect.getmembers(torchvision.models):
        if model_name == member[0] and type(member[1]) == types.FunctionType:
            model = member[1](pretrained=pretrained, progress=False)
            image_size = 299 if model_name == 'inception_v3' else 224
            aux = None

            if model_name.startswith('alexnet') or model_name.startswith('vgg'):
                fv_dim = model.classifier[6].in_features
                head = nn.Linear(fv_dim, num_classes)
                model.classifier[6] = head

            elif model_name.startswith('resnet') or model_name.startswith('shufflenet'):
                fv_dim = model.fc.in_features
                head = nn.Linear(fv_dim, num_classes)
                model.fc = head

            elif model_name.startswith('mobilenet') or model_name.startswith('mnasnet'):
                fv_dim = model.classifier[1].in_features
                head = nn.Linear(fv_dim, num_classes)
                model.classifier[1] = head

            elif model_name.startswith('squeezenet'):
                model.num_classes = num_classes
                fv_dim = 512
                head = nn.Conv2d(fv_dim, num_classes, kernel_size=(1, 1), stride=(1, 1))
                model.classifier[1] = head

            elif model_name.startswith('densenet'):
                fv_dim = model.classifier.in_features
                head = nn.Linear(fv_dim, num_classes)
                model.classifier = head

            elif model_name.startswith('inception'):
                head_layers = []
                # Handle the primary net
                fv_dim = model.fc.in_features
                primary_layer = nn.Linear(fv_dim, num_classes)
                model.fc = primary_layer
                head_layers.append(primary_layer)
                # Handle the auxilary net
                aux_fv_dim = model.AuxLogits.fc.in_features
                aux_layer = nn.Linear(aux_fv_dim, num_classes)
                model.AuxLogits.fc = aux_layer
                head_layers.append(aux_layer)

                head = nn.ModuleList(head_layers)
                aux = nn.ModuleList([model.AuxLogits])

            elif model_name.startswith('googlenet'):
                model = torchvision.models.googlenet(pretrained=pretrained, progress=False, aux_logits=True)
                head_layers = []
                aux_layers = []
                # Handle the primary net
                fv_dim = model.fc.in_features
                primary_layer = nn.Linear(fv_dim, num_classes)
                model.fc = primary_layer
                head_layers.append(primary_layer)
                # Handle the auxilary nets
                for module in [model.aux1, model.aux2]:
                    aux_fv_dim = module.fc2.in_features
                    aux_layer = nn.Linear(aux_fv_dim, num_classes)
                    module.fc2 = aux_layer
                    head_layers.append(aux_layer)
                    aux_layers.append(module)
                head = nn.ModuleList(head_layers)
                aux = nn.ModuleList(aux_layers)

            else:
                raise NotImplementedError

            if type(head) == nn.ModuleList:
                for layer in head:
                    nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.xavier_uniform_(head.weight)

            return model, head, aux, image_size, fv_dim

    raise KeyError('Unknown model. Please select one from {}'.format(list_available_models()))


'''
alexnet
densenet121
densenet161
densenet169
densenet201
googlenet
inception_v3
mnasnet0_5
mnasnet0_75
mnasnet1_0
mnasnet1_3
mobilenet_v2
resnet101
resnet152
resnet18
resnet34
resnet50
resnext101_32x8d
resnext50_32x4d
shufflenet_v2_x0_5
shufflenet_v2_x1_0
shufflenet_v2_x1_5
shufflenet_v2_x2_0
squeezenet1_0
squeezenet1_1
vgg11
vgg11_bn
vgg13
vgg13_bn
vgg16
vgg16_bn
vgg19
vgg19_bn
wide_resnet101_2
wide_resnet50_2
'''