import torch
from transformers import set_seed


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def set_launch_seed(seed: int):
    if seed is not None:
        set_seed(seed)


def freeze_model_parameters(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
