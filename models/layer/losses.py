# -*- coding: utf-8 -*-
"""Module containing a variety of loss functions for complex valued neural networks."""

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import torch.nn as nn
import utils.fourier as FFT
import utils.transforms as transforms


class log_l1(nn.Module):
    """
    The log_l1 loss function.

    This loss function is a combination of L1 loss and logarithmic L1 loss.

    Args:
        alpha (float): A weight factor to balance the contribution of L1 loss and logarithmic L1 loss.

    Attributes:
        alpha (float): A weight factor to balance the contribution of L1 loss and logarithmic L1 loss.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initializes the log_l1 loss function.

        Args:
            alpha (float): A weight factor to balance the contribution of L1 loss and logarithmic L1 loss.
        """
        super(log_l1, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred: torch.cfloat, y: torch.cfloat) -> torch.cfloat:
        """
        Calculates the log_l1 loss.

        Args:
            y_pred (torch.Tensor): Prediction tensor of shape (batch_size, ...).
            y (torch.Tensor): Target tensor of shape (batch_size, ...).

        Returns:
            torch.Tensor: log_l1 loss.
        """
        l1 = nn.L1Loss()
        loss = (
            l1(y_pred, y)
            + self.alpha * l1(torch.log(y_pred + 1e-6), torch.log(y + 1e-6))
        ) / 2  # add +1e-6 as log(0) is undefined

        return loss


class l1_imagespace(nn.Module):
    """A module implementing the L1 loss in image space.

    This module takes as input the predicted values `y_pred` and the ground truth `y` in the
    image space, and computes the L1 loss between them.

    Attributes:
        l1 (nn.L1Loss): The L1 loss function.

    Returns:
        torch.cfloat: The L1 loss between the input tensors.
    """

    def __init__(self):
        """
        Initializes the l1_imagespace loss module.
        """
        super(l1_imagespace, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, y_pred: torch.cfloat, y: torch.cfloat):
        """Compute the L1 loss between two image space tensors.

        Args:
            y_pred (torch.cfloat): The predicted image tensor.
            y (torch.cfloat): The ground truth image tensor.

        Returns:
            torch.cfloat: The L1 loss between the two tensors.
        """

        # Transform complex input to image space
        y, y_pred = FFT.ifft(y), FFT.ifft(y_pred)

        # Compute L1 loss
        loss = self.l1(y_pred, y)

        return loss


class L1_pha_abs(nn.Module):
    def __init__(self) -> None:
        super(L1_pha_abs, self).__init__()

        self.l1 = nn.L1Loss()

    def forward(self, y_pred: torch.cfloat, y: torch.cfloat):
        loss_mag = self.l1(y_pred.abs(), y.abs())
        loss_ang = self.l1(y_pred.angle(), y.angle())

        return (loss_mag + loss_ang) / 2


class MSELoss(nn.Module):
    """Implementation of Mean Squared Error loss for complex numbers.

    This class computes the Mean Squared Error loss between two complex numbers.
    It calculates the loss as the sum of mean squared error between the magnitude and
    phase of the output and target.

    Args:
        nn.Module: Inherited nn.Module class.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        """Compute the Mean Squared Error loss.

        This method computes the Mean Squared Error loss between two complex numbers.
        The loss is calculated as the sum of mean squared error between the real
        and imaginary parts of the output and target.

        Args:
            output (torch.Tensor): Complex valued outputs from the neural network with shape (batch_size, ...).
            target (torch.Tensor): Complex valued targets with shape (batch_size, ...).

        Returns:
            torch.Tensor: the computed loss.
        """
        if output.dtype == torch.cfloat:
            loss = self.mse(output.real, target.real) + self.mse(
                output.imag, target.imag
            )
        else:
            loss = self.mse(output, target)

        return loss


class PSNR(nn.Module):
    """
    Class to calculate the Peak Signal-to-Noise Ratio (PSNR) between a models output and the ground truth.

    Attributes:
        mse (nn.MSELoss): An instance of PyTorch's Mean Squared Error (MSE) loss function.

    Methods:
        forward(output, target): Computes the PSNR between two tensors.
    """

    def __init__(self) -> None:
        super(PSNR, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        """
        Calculates the PSNR between two images.

        Args:
            output (torch.Tensor): Output tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            float: PSNR value.
        """
        psnr = 10 * torch.log10(
            output.abs().max() ** 2 / (self.mse(target.abs(), output.abs()))
        )
        return psnr


class NMSELoss(nn.Module):
    """
    Class to calculate the Normalized Mean Squared Error (NMSE) between two tensors.

    Methods:
        forward(output, target): Computes the NMSE between two tensors.

    """

    def __init__(self) -> None:
        super(NMSELoss, self).__init__()

    def forward(self, output, target):
        """
        Calculates the NMSE between two tensors in image-space.

        Args:
            output (torch.Tensor): Output tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            float: NMSE value.
        """
        output, target = FFT.ifft(output), FFT.ifft(target)

        return (((output - target) / target) ** 2).mean().abs()


class MSEimagLoss(nn.Module):
    """Implementation of the mean squared error loss for complex numbers.

    This module calculates the mean squared error loss between the
    complex valued output and target. The output and target are first
    transformed using the inverse fast Fourier transform (IFFT).

    Args:
        nn.Module: Inherited nn.Module class.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """Computes the mean squared error loss between the complex valued output and target.

        Args:
            output (torch.Tensor): Complex valued outputs from the neural network with shape (batch_size, ...).
            target (torch.Tensor): Complex valued targets with shape (batch_size, ...).

        Returns:
            torch.Tensor: The absolute mean squared error loss between the imaginary parts of output and target.
        """

        output = FFT.ifft(output)
        target = FFT.ifft(target)

        return ((output - target) ** 2).mean().abs()


class MSELogLoss(nn.Module):
    """Mean Squared Error Loss for Log Magnitude and Phase Spectrogram.

    This module implements the Mean Squared Error Loss for the log magnitude and phase spectrogram. The loss
    function is calculated as the mean squared error of the log magnitude and log phase.

    Args:
        nn.Module: Inherited nn.Module class.

    Attributes:
        mse (nn.MSELoss): Mean squared error loss.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        """Computes the mean squared error loss of log magnitude and phase spectrogram.

        Args:
            output (torch.Tensor): Complex valued outputs from the neural network with shape (batch_size, ...).
            target (torch.Tensor): Complex valued targets with shape (batch_size, ...).

        Returns:
            float: Mean squared error loss for log magnitude and log phase.
        """
        log_output = torch.log(output + 1e-8)
        log_target = torch.log(target + 1e-8)

        return self.mse(log_output.abs(), log_target.abs()) + self.mse(
            log_output.angle(), log_target.angle()
        )


class L1NormAbsLoss(nn.Module):
    """Calculates the L1 loss between normalized absolute values of outputs and targets.

    This module takes as input the outputs and targets from a neural network and normalizes
    their absolute values, then calculates the L1 loss between the two. The normalization
    is done by subtracting the minimum value and dividing by the range of the absolute values.
    The loss is calculated using the PyTorch L1Loss module.

    Args:
        nn.Module: Inherited nn.Module class.

    Returns:
        torch.Tensor: A scalar representing the L1 loss between the normalized absolute values of the outputs
        and targets.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        """Calculates the L1 loss between normalized absolute values of outputs and targets.

        Args:
            output (torch.Tensor): Complex valued outputs from the neural network with shape (batch_size, ...).
            target (torch.Tensor): Complex valued targets with shape (batch_size, ...).

        Returns:
            torch.Tensor: A scalar representing the L1 loss between the normalized absolute values of the outputs
            and targets.
        """
        output_norm = transforms.tensor_normalization(output)
        target_norm = transforms.tensor_normalization(target)

        return self.l1(output_norm, target_norm)
    

class CombinedLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):

        kspace_loss = self.l1(output, target)
        image_loss = self.l1(FFT.ifft(output), FFT.ifft(target))

        return kspace_loss + image_loss * 100