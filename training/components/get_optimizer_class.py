import inspect
import torch.optim as optim


def list_available_optimizers():
    names = []
    for member in inspect.getmembers(optim):
        if type(member[1]) == type:
            if member[0] != 'Optimizer':
                names.append(member[0])
    return names


def get_optimizer_class(optimizer_name):
    for member in inspect.getmembers(optim):
        if type(member[1]) == type:
            if member[0] == optimizer_name:
                return member[1]
    raise KeyError(f'No optimizer {optimizer_name} found. Use one of the following:{str(list_available_optimizers())}')


'''
ASGD
Adadelta
Adagrad
Adam
AdamW
Adamax
LBFGS
RMSprop
Rprop
SGD
SparseAdam
'''
