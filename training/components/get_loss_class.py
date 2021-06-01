import inspect
import torch.nn as nn


def list_available_losses():
    names = []
    for member in inspect.getmembers(nn):
        if member[0].endswith('Loss'):
            names.append(member[0])
    return names


def get_loss_class(criterion_name):
    require_one_hot = criterion_name != 'CrossEntropyLoss' and criterion_name != 'NLLLoss'
    for member in inspect.getmembers(nn):
        if member[0].endswith('Loss'):
            if member[0] == criterion_name:
                return member[1], require_one_hot
    raise KeyError(f'No criterion function {criterion_name} found. Use one of the :{str(list_available_losses())}')


'''
AdaptiveLogSoftmaxWithLoss
BCELoss
BCEWithLogitsLoss
CTCLoss
CosineEmbeddingLoss
CrossEntropyLoss
HingeEmbeddingLoss
KLDivLoss
L1Loss
MSELoss
MarginRankingLoss
MultiLabelMarginLoss
MultiLabelSoftMarginLoss
MultiMarginLoss
NLLLoss
PoissonNLLLoss
SmoothL1Loss
SoftMarginLoss
TripletMarginLoss
'''