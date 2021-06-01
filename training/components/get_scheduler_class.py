import inspect
import torch.optim.lr_scheduler as lr_scheduler


def list_available_schedulers():
    names = []
    for member in inspect.getmembers(lr_scheduler):
        if type(member[1]) == type:
            if member[0] != 'Counter' and member[0] != 'Optimizer' and member[0] != '_LRScheduler':
                names.append(member[0])
    return names


def get_scheduler_class(lr_scheduler_name):
    for member in inspect.getmembers(lr_scheduler):
        if type(member[1]) == type:
            if member[0] == lr_scheduler_name:
                return member[1]
    raise KeyError(f'No scheduler {lr_scheduler} found. Use one of the following:{list_available_schedulers()}')


'''
CosineAnnealingLR
CosineAnnealingWarmRestarts
CyclicLR
ExponentialLR
LambdaLR
MultiStepLR
MultiplicativeLR
OneCycleLR
ReduceLROnPlateau
StepLR
'''