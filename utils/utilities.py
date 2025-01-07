# -*- coding: utf-8 -*-
import torch
import functools
import time
import logging
import torch.nn.functional as F
from prettytable import PrettyTable

def count_parameters(model: torch.nn.Module) -> tuple[PrettyTable, int]:
    """Counts the model parameters

    Args:2
        model (torch.nn.Module): a torch model

    Returns:
        int: number of model parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    return table, total_params


def timer(func):
    """Decorator that measures the runtime of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        wrapper_timer (function): The decorated function.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(f"Finished {func.__name__!r} in {run_time:.3f} secs")
        return value

    return wrapper_timer

def padding(pooled_input, original):
    """Pad a pooled input tensor to match the size of an original tensor.

    This function pads the 'pooled_input' tensor to match the spatial dimensions
    (height and width) of the 'original' tensor. It calculates the amount of padding
    required on each side and applies it symmetrically.

    Args:
        pooled_input (torch.Tensor): The pooled input tensor to be padded.
        original (torch.Tensor): The original tensor whose spatial dimensions
            the 'pooled_input' tensor should match.

    Returns:
        torch.Tensor: The padded 'pooled_input' tensor with the same spatial
        dimensions as the 'original' tensor.
    """

    pad_h = original.size(2) - pooled_input.size(2)
    pad_w = original.size(3) - pooled_input.size(3)
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    padded = F.pad(pooled_input, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom))

    return padded