# -*- coding: utf-8 -*-
"""Module containing a variety of activation functions for complex valued neural networks."""

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fourier import ifft, fft


class ComplexELU(nn.Module):
    """Applies the exponential linear unit (ELU) activation function element-wise to the real and imaginary parts
    of a complex input tensor.

    Args:
        inplace (bool, optional): If set to True, the input tensor is modified in-place, i.e. without allocating any
            new memory. Otherwise, a new tensor with the same shape as the input tensor is returned. Defaults to False.

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the ComplexELU activation function.
    """

    def __init__(self, inplace=False):
        """
        Initializes the ComplexELU activation function.

        Args:
            inplace (bool, optional): If set to True, the input tensor is modified in-place.
                Defaults to False.
        """
        super(ComplexELU, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the ComplexELU activation function.

        Args:
            x (torch.Tensor): The complex-valued input tensor.

        Returns:
            torch.Tensor: The complex-valued output tensor after applying the ComplexELU activation function.
        """
        if self.inplace:
            output = F.elu(x.real, inplace=True) + 1j * F.elu(x.imag, inplace=True)
        else:
            output = F.elu(x.real) + 1j * F.elu(x.imag)

        return output


class ComplexReLU(nn.Module):
    """
    Applies ReLU to real and imaginary parts independently.

    Args:
        inplace (bool, optional): Performs the operation in-place. Default is False.

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the ComplexReLU activation function.
    """

    def __init__(self, inplace=False):
        """
        Initializes ComplexReLU.

        Args:
            inplace (bool, optional): Performs the operation in-place. Default is False.
        """
        super(ComplexReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of ComplexReLU.

        Args:
            x (torch.Tensor): Input tensor with complex numbers.

        Returns:
            torch.Tensor: Output tensor with ReLU applied to real and imaginary parts.
        """
        if self.inplace:
            output = torch.real(x).relu_() + 1j * torch.imag(x).relu_()
        else:
            output = torch.real(x).relu() + 1j * torch.imag(x).relu()

        return output


class ComplexLReLU(nn.Module):
    """
    Applies Leaky ReLU to real and imaginary parts independently.

    Args:
        inplace (bool, optional): Performs the operation in-place. Default is False.
    """

    def __init__(self, inplace=False):
        """
        Initializes the Leaky ReLU module.

        Args:
            inplace (bool, optional): Performs the operation in-place. Default is False.
        """
        super(ComplexLReLU, self).__init__()
        self.inplace = inplace
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of Leaky ReLU.

        Args:
            x (torch.Tensor): Input tensor with complex numbers.

        Returns:
            torch.Tensor: Output tensor with Leaky ReLU applied to real and imaginary parts.
        """
        out = self.leaky_relu(x.real) + 1j * self.leaky_relu(x.imag)

        return out


class ComplexPReLU(nn.Module):
    """
    Applies Parametric ReLU (PReLU) to real and imaginary parts independently.

    This module consists of separate PReLU activations for the real and imaginary parts
    of complex numbers.

    Args:
        None

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the ComplexPReLU activation function.
    """

    def __init__(self) -> None:
        """
        Initializes ComplexPReLU.

        Args:
            None
        """
        super(ComplexPReLU, self).__init__()

        self.prelu_real = nn.PReLU()
        self.prelu_imag = nn.PReLU()

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of ComplexPReLU.

        Args:
            x (torch.Tensor): Input tensor with complex numbers.

        Returns:
            torch.Tensor: Output tensor with PReLU applied to real and imaginary parts.
        """
        out = self.prelu_real(x.real) + 1j * self.prelu_imag(x.imag)
        return out


class ComplexSELU(nn.Module):
    """Applies the Scaled Exponential Linear Unit (SELU) activation function to complex-valued input.

    Args:
        inplace (bool, optional): If True, modifies the input tensor in-place. Default is False.

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the SELU activation function.
    """

    def __init__(self, inplace=False) -> None:
        """Initializes ComplexSELU.

        Args:
            inplace (bool, optional): If True, modifies the input tensor in-place. Default is False.
        """
        super(ComplexSELU, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the ComplexSELU layer.

        Args:
            x (torch.Tensor): A complex-valued tensor to be processed by the ComplexSELU layer.

        Returns:
            torch.Tensor: The complex-valued output tensor after applying the SELU activation function.
        """
        if self.inplace:
            output = torch.selu_(x.real) + 1j * torch.selu_(x.imag)
        else:
            output = torch.selu(x.real) + 1j * torch.selu(x.imag)

        return output


class ComplexTanh(nn.Module):
    """
    Applies the hyperbolic tangent to a complex number.

    The hyperbolic tangent of a complex number z is defined as:
    tanh(z) = (tanh(Re(z)) + i * tan(Im(z))) / (1 + i * tanh(Re(z)) * tan(Im(z)))

    Args:
        None

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the ComplexTanh activation function.
    """

    def __init__(self) -> None:
        """
        Initializes ComplexTanh.

        Args:
            None
        """
        super(ComplexTanh, self).__init__()

    def forward(self, x):
        """
        Forward pass of ComplexTanh.

        Args:
            x (torch.Tensor): Input tensor with complex numbers.

        Returns:
            torch.Tensor: Output tensor with hyperbolic tangent applied to real and imaginary parts.
        """
        x = nn.Parameter(
            (torch.tanh(x.real) + 1j * torch.tan(x.imag))
            / (1 + 1j * torch.tanh(x.real) * torch.tan(x.imag))
        )

        return x


class ComplexSigmoid(nn.Module):
    """Applies the sigmoid activation function to complex-valued input.

    Args:
        None

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the sigmoid activation function.
    """

    def __init__(self) -> None:
        """Initializes ComplexSigmoid.

        Args:
            None
        """
        super(ComplexSigmoid, self).__init__()

    def forward(self, x: torch.cfloat) -> torch.cfloat:
        """Forward pass of the ComplexSigmoid layer.

        Args:
            x (torch.Tensor): A complex-valued tensor to be processed by the ComplexSigmoid layer.

        Returns:
            torch.Tensor: The complex-valued output tensor after applying the sigmoid activation function.
        """
        x = torch.sigmoid(x.real) + 1j * torch.sigmoid(x.imag)

        return x.cfloat()


class PhaseAmplitudeReLU(nn.Module):
    """Applies the Rectified Linear Unit (ReLU) activation function to the amplitude and phase of complex-valued input.

    Args:
        inplace (bool, optional): If True, modifies the input tensor in-place. Default is False.

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the Phase Amplitude ReLU activation function.
    """

    def __init__(self, inplace=False) -> None:
        """Initializes PhaseAmplitudeReLU.

        Args:
            inplace (bool, optional): If True, modifies the input tensor in-place. Default is False.
        """
        super(PhaseAmplitudeReLU, self).__init__()

    def forward(self, x: torch.cfloat) -> torch.cfloat:
        """Forward pass of the PhaseAmplitudeReLU layer.

        Args:
            x (torch.Tensor): A complex-valued tensor to be processed by the PhaseAmplitudeReLU layer.

        Returns:
            torch.Tensor: The complex-valued output tensor after applying the Phase Amplitude ReLU activation function.
        """
        return torch.relu(x.abs()) * torch.exp(1.0j * torch.relu(x.angle()))


class PhaseReLU(nn.Module):
    """Applies the Rectified Linear Unit (ReLU) activation function to the phase of complex-valued input.

    Args:
        None

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the Phase ReLU activation function.
    """

    def __init__(self) -> None:
        """Initializes PhaseReLU.

        Args:
            None
        """
        super(PhaseReLU, self).__init__()

    def forward(self, x: torch.cfloat) -> torch.cfloat:
        """Forward pass of the PhaseReLU layer.

        Args:
            x (torch.Tensor): A complex-valued tensor to be processed by the PhaseReLU layer.

        Returns:
            torch.Tensor: The complex-valued output tensor after applying the Phase ReLU activation function.
        """
        return torch.abs(x) * (
            torch.cos(F.relu(torch.angle(x))) + 1j * torch.sin(F.relu(torch.angle(x)))
        )


class ComplexCardioid(nn.Module):
    """Applies the complex cardioid activation function to complex-valued input.

    The complex cardioid function is defined as f(z) = 0.5 * (1 + cos(ang(z))) * z, where ang(z) is the angle of z.

    Args:
        None

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the Complex Cardioid activation function.
    """

    def __init__(self) -> None:
        """Initializes ComplexCardioid.

        Args:
            None
        """
        super(ComplexCardioid, self).__init__()

    def forward(self, x: torch.cfloat) -> torch.cfloat:
        """Forward pass of the ComplexCardioid layer.

        Args:
            x (torch.Tensor): A complex-valued tensor to be processed by the ComplexCardioid layer.

        Returns:
            torch.Tensor: The complex-valued output tensor after applying the Complex Cardioid activation function.
        """
        output = 0.5 * (1 + torch.cos(x.angle())) * x

        return output


class AmplitudeRelu(nn.Module):
    """Applies the amplitude rectified linear unit (Amplitude ReLU) activation function to complex-valued input.

    The Amplitude ReLU function is defined as f(z) = max(0, |z|) * exp(j * max(0, ang(z) + bias)),
    where |z| is the magnitude of z, ang(z) is the angle of z, and 'bias' is a learnable parameter.

    Args:
        None

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the Amplitude ReLU activation function.
    """

    def __init__(self) -> None:
        """Initializes AmplitudeRelu.

        Args:
            None
        """
        super(AmplitudeRelu, self).__init__()
        self.bias = nn.Parameter(torch.Tensor([torch.pi]), requires_grad=True)

    def forward(self, x: torch.cfloat) -> torch.cfloat:
        """Forward pass of the AmplitudeRelu layer.

        Args:
            x (torch.Tensor): A complex-valued tensor to be processed by the AmplitudeRelu layer.

        Returns:
            torch.Tensor: The complex-valued output tensor after applying the Amplitude ReLU activation function.
        """
        return F.leaky_relu(x.abs()) * torch.exp(
            1j * F.leaky_relu(x.angle() + self.bias)
        )


class cLogReLU(nn.Module):
    """Applies the complex logarithm of the rectified linear unit (ReLU) activation function to complex-valued input.

    The cLogReLU function is defined as f(z) = log(ReLU(z)), where ReLU(z) is the rectified linear unit applied to
    each element of the complex input z.

    Args:
        None

    Returns:
        torch.Tensor: The complex-valued output tensor after applying the cLogReLU activation function.
    """

    def __init__(self) -> None:
        """Initializes cLogReLU.

        Args:
            None
        """
        super(cLogReLU, self).__init__()
        self.relu = ComplexReLU()

    def forward(self, x: torch.cfloat) -> torch.cfloat:
        """Forward pass of the cLogReLU layer.

        Args:
            x (torch.Tensor): A complex-valued tensor to be processed by the cLogReLU layer.

        Returns:
            torch.Tensor: The complex-valued output tensor after applying the cLogReLU activation function.
        """
        return torch.log(self.relu(x))
    

class PhaseSensitiveReLU(nn.Module):
    def __init__(self, threshold=0.1):
        super(PhaseSensitiveReLU, self).__init__()
        self.threshold = threshold
        
    def forward(self, x):
        #magnitude = torch.abs(x)
        #phase = torch.angle(x)
        #activated_magnitude = torch.where(magnitude > self.threshold, magnitude, torch.zeros_like(magnitude))
        #return activated_magnitude * torch.exp(1j * phase)
        return x

class SineReLU(nn.Module):
    def __init__(self):
        super(SineReLU, self).__init__()
        
    def forward(self, x):
        return torch.sin(x.real) + 1j * torch.sin(x.imag)
    
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        
    def forward(self, x):
        return x