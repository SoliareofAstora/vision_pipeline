from model.torch_cnn_classifier.get_torch_model import list_available_models
from model.torch_cnn_classifier.torch_cnn import TorchCNN


def get_model_class(model_name):
    if model_name in list_available_models():
        return TorchCNN
    if model_name == "mil":
        raise NotImplementedError

    raise KeyError
