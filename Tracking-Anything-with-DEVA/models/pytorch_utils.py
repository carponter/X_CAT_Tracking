import torch
import numpy as np

device = None

def zeros(*sizes, torch_device=None):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes).to(torch_device)

def zeros_like(x, torch_device=None):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(x).to(torch_device)

def ones(*sizes, torch_device=None):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes).to(torch_device)

def ones_like(x, torch_device=None):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(x).to(torch_device)

def randn(*sizes, torch_device=None):
    if torch_device is None:
        torch_device = device
    return torch.randn(*sizes).to(torch_device)

def randn_like(x, torch_device=None):
    if torch_device is None:
        torch_device = device
    return torch.randn_like(x).to(torch_device)

def tensor(x, torch_device=None):
    if torch_device is None:
        torch_device = device
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(torch_device)
    return torch.tensor(x, device=torch_device)