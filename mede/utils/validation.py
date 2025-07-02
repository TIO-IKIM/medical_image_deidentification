# -*- coding: utf-8 -*-
import torch
import torchmetrics.functional as f
import torch.nn.functional as F
import matplotlib.pyplot as plt


def segmentation_validation(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Perform validation for segmentation predictions.

    Args:
        predictions (torch.Tensor): Predicted segmentation masks.
        targets (torch.Tensor): Ground truth segmentation masks.

    Returns:
        dict: Dictionary containing validation metrics.
            - 'dsc': Dice similarity coefficient.
            - 'iou': Intersection over Union.
    """

    val_metrics = {}

    predictions = torch.sigmoid(predictions)

    val_metrics["dsc"] = f.dice(predictions.float(), targets.int(), ignore_index=0)
    val_metrics["iou"] = f.jaccard_index(
        predictions.float(), targets.int(), ignore_index=0, task="binary"
    )

    return val_metrics


def plot_segmentation(
    model_input: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    output_path: str,
) -> None:
    """
    Plot the input samples, targets, and predictions for segmentation.

    Args:
        model_input (Tensor): Input samples.
        predictions (Tensor): Predicted segmentation masks.
        targets (Tensor): Target segmentation masks.
        epoch (int): Current epoch number.
        output_path (str): Path to save the plots.

    Returns:
        None
    """

    predictions = torch.sigmoid(predictions) > 0.5
    predictions = predictions * model_input

    plt.imsave(f"{output_path}/pred_epoch{epoch}.png", predictions[0,0,30,...].cpu())
    plt.imsave(f"{output_path}/target_epoch{epoch}.png", targets[0,0,30,...].cpu())