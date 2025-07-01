# -*- coding: utf-8 -*-
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
import torch


class DiceLoss(nn.Module):
    """
    Dice Loss implementation for binary segmentation tasks.
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(
        inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1
    ) -> torch.Tensor:
        """
        Compute the Dice Loss between the predicted inputs and the target labels.

        Args:
            inputs (Tensor): The predicted inputs from the model.
            targets (Tensor): The target labels.
            smooth (float, optional): A smoothing factor to avoid division by zero. Default: 1

        Returns:
            Tensor: The computed Dice Loss.
        """
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1).long()

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss function for binary segmentation.

    Args:
        alpha (int): The balancing factor for the positive class in the loss calculation. Default is 1.
        gamma (int): The focusing parameter that controls the rate at which easy examples are down-weighted. Default is 2.
        num_classes (int): The number of classes in the segmentation task. Default is 2.
        reduction (str): The reduction method to apply to the computed loss. Default is "sum".

    """

    def __init__(
        self,
        alpha: int = 1,
        gamma: int = 2,
        num_classes: int = 2,
        reduction: str = "sum",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss for binary segmentation.

        Args:
            inputs (Tensor): The input tensor containing the predicted logits.
            targets (Tensor): The target tensor containing the ground truth labels.

        Returns:
            Tensor: The computed focal loss.

        """
        loss = sigmoid_focal_loss(
            inputs,
            targets.float(),
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss
