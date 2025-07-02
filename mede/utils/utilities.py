# -*- coding: utf-8 -*-
import torch
import functools
import time
import logging
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
