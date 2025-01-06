# -*- coding: utf-8 -*-
"""Module containing a variety of layer for complex valued neural networks."""

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import torch.nn as nn
import utils.utilities as u
from models.layer.activations import ComplexReLU
import models.layer.spectral_layer as S


class ComplexDoubleConv(nn.Module):
    """
    A double convolutional layer in PyTorch with support for complex numbers and various activation functions.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): Size of the convolving kernel.
        device: The device on which the module is run.
        padding (int): Zero-padding added to both sides of the input.
        dropout (float): Dropout probability. Defaults to 0.0.
        dim (int): Dimension of the input data. 2 for 2D slices, 3 for 3D volumes. Defaults to 2.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        activation (nn.Module, optional): The activation function to use. Defaults to ComplexReLU().
        logger (object, optional): A logging object. Defaults to None.

    Attributes:
        conv (nn.Sequential): A sequential module that performs the 2D complex convolution, batch normalization,
            and activation operations.
        activation (torch.nn.Module): The activation function used in the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        device,
        padding: int,
        dropout: float = 0.0,
        dim: int = 2,
        stride: int = 1,
        dilation: int = 1,
        activation: nn.Module = ComplexReLU(),
    ):
        super(ComplexDoubleConv, self).__init__()
        # Use torch.jit.script() for all layers in nn.Sequential to implement torchscript to reduce computation times.
        # Be aware of much higher memory consumption!

        self.activation = activation
        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d

        self.conv = nn.Sequential(
            Conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
                device=device,
                dtype=torch.complex64,
            ),
            S.ComplexNaiveBatchnorm(out_channels, dim=dim),
            self.activation,
            Conv(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
                device=device,
                dtype=torch.complex64,
            ),
            S.ComplexNaiveBatchnorm(out_channels, dim=dim),
            self.activation,
            S.ComplexDropout(p=dropout),
        )

    def forward(self, x):
        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)

        out = self.conv(x)

        if out.ndim == 5:
            out = out.permute(0, 1, 3, 4, 2)

        return out


class ComplexConv2d(nn.Module):
    """
    A complex-valued convolutional layer for 2D inputs.

    This layer implements a complex convolutional operation on the input tensor, treating the real and
    imaginary parts separately. It consists of two conventional 2D convolutional layers, one for the
    real part and another for the imaginary part. The complex convolution is performed by subtracting and
    adding the outputs of these two convolutional layers, respectively.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolutional operation.
        padding (int or tuple): Padding added to all sides of the input.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to False.

    Returns:
        torch.Tensor: The complex-valued output tensor after passing through the convolutional layer.

    Note:
        The complex convolution operation is defined as follows:
        If x = x_real + 1j * x_imag and w = w_real + 1j * w_imag, then
        complex_conv(x, w) = (x_real * w_real - x_imag * w_imag) + 1j * (x_real * w_imag + x_imag * w_real)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias: bool = False,
    ):
        super(ComplexConv2d, self).__init__()

        # Two separate convolutional layers for real and imaginary parts
        self.conv_re = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv_im = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        """
        Forward pass of the ComplexConv2d layer.

        Args:
            x (torch.Tensor): Input complex-valued tensor.

        Returns:
            torch.Tensor: The complex-valued output tensor after passing through the convolutional layer.
        """
        # Separate real and imaginary parts of the input
        x_real = x.real
        x_imag = x.imag

        # Complex convolution operation
        real = self.conv_re(x_real) - self.conv_im(x_imag)
        imag = self.conv_re(x_imag) + self.conv_im(x_real)

        # Combine real and imaginary parts back into a complex tensor
        return u.to_complex(real, imag)


class ComplexConvTranspose(nn.Module):
    """
    Complex-valued Convolutional Transpose Layer.

    This layer performs the complex-valued transposed convolution operation.
    It consists of separate convolution operations for real and imaginary parts,
    which are then combined.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolution kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
        dim (int, optional): Dimension of the convolution operation (2 or 3). Defaults to 2.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to False.

    Returns:
        torch.cfloat: The complex-valued output tensor after applying the convolution transpose.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dim: int = 2,
        bias: bool = False,
    ):
        super(ComplexConvTranspose, self).__init__()
        if dim == 2:
            ConvTranspose = nn.ConvTranspose2d
        elif dim == 3:
            ConvTranspose = nn.ConvTranspose3d

        self.conv_re = ConvTranspose(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv_im = ConvTranspose(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)

        # ConvTranspose3d creates handful of NaN values per volume. Set these to zero.
        output = (
            torch.nan_to_num(self.conv_re(x.real) - self.conv_im(x.imag), nan=0)
            + 1j
            * torch.nan_to_num(
                self.conv_re(x.imag) + self.conv_im(x.real), nan=0
            ).cfloat()
        )

        if output.ndim == 5:
            output = output.permute(0, 1, 3, 4, 2)

        return output


class ResidualBlock(nn.Module):
    """Implementation of a residual block for use in a neural network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        device (torch.device): The device to run on.
        kernel_size (int): Size of the convolutional kernel.
        dim (int): Dimension of the input data. 2 for 2D slices, 3 for 3D volumes. Defaults to 2.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dropout (float): Dropout probability. Defaults to 0.0.
        padding (int, optional): Zero-padding added to both sides of the input. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        activation (nn.Module, optional): Type of activation function. Defaults to ComplexReLU().
        resample (function, optional): A function to resample the input. Defaults to None.

    Attr:
        conv1 (nn.Sequential): The first convolutional layer of the residual block.
        conv2 (nn.Sequential): The second convolutional layer of the residual block.
        activation (nn.Module): The activation function used in the residual block.
        out_channels (int): Number of output channels.
        residual (nn.Sequential): Residual layer. If `downsample` is True, apply a convolution to downsample the input.
            Otherwise the layer is a simple linear layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        device,
        kernel_size: int,
        dim: int = 2,
        stride: int = 1,
        dropout: float = 0.0,
        padding: int = 1,
        dilation: int = 1,
        activation: nn.Module = ComplexReLU(),
        resample=False,
    ) -> None:
        super(ResidualBlock, self).__init__()

        self.activation = activation()
        if dim == 2:
            Conv = nn.Conv2d
            #Conv = S.FFTConv
        elif dim == 3:
            Conv = nn.Conv3d

        # downsampling here means an increase in feature maps, as we want to keep the image size
        if resample:
            self.residual = nn.Sequential(
                Conv(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                    device=device,
                    dtype=torch.complex64,
                ),
            )
        else:
            self.residual = nn.Sequential()
        self.conv1 = nn.Sequential(
            Conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
                device=device,
                dtype=torch.complex64,
            ),
            S.ComplexNaiveBatchnorm(num_features=out_channels, dim=dim),
            activation(),
            S.ComplexDropout(p=dropout),
        )
        self.conv2 = nn.Sequential(
            Conv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
                device=device,
                dtype=torch.complex64,
            ),
            S.ComplexNaiveBatchnorm(out_channels, dim=dim),
            S.ComplexDropout(p=dropout),
        )
        self.out_channels = out_channels

    def forward(self, x):
        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)

        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)

        if out.ndim == 5:
            out = out.permute(0, 1, 3, 4, 2)

        return out


class Upsampling(nn.Module):
    """Upsampling module for a decoder using nearerst neighbour upsampling followed by a convolution layer.

    The input is first upsampled by the `scale_factor` and then given to a convolution layer.
    The complex valued input is split into real and imaginary part for the upsampling and concateneted
    to a complex tensor again before the convolution layer.


    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        device (torch.device): The device to run on.
        scale_factor (int): Scaling factor of the input for the upsampling layer.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Zero-padding added to both sides of the input. Defaults to 1.
        dim (int, optional): Dimension of the input. Defaults to 2.

    Attr:
        upsampling (nn.Module): Upsampling module which scales the input by the `scale_factor`.
        conv (nn.Module): Convolution layer following the upsampling layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        device,
        scale_factor: int,
        kernel_size: int = 3,
        padding: int = 1,
        dim: int = 2,
    ):
        super(Upsampling, self).__init__()
        #self.upsampling = nn.Upsample(scale_factor=scale_factor)
        self.upsampling = S.SpectralUpsampling()
        if dim == 2:
            Conv = nn.Conv2d
            #Conv = S.FFTConv
        elif dim == 3:
            Conv = nn.Conv3d

        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
            device=device,
            dtype=torch.complex64,
        )

    def forward(self, x: torch.Tensor, cut_off: torch.Tensor = None) -> torch.Tensor:
        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)

        x = self.upsampling(input=x, cut_off=cut_off)
        x = self.conv(x)

        return x
