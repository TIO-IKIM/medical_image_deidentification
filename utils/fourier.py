# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch


@torch.jit.script
def ifft(scan: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:
    """Calculate the inverse fourier transform of an complex valued input tensor
    in the k-space (frequency domain) to transform it into image space.

    Use the inverse fast fourier transform (iFFT) module `torch.fft.ifft2` (if 4D tensor)
    or `torch.fft.ifftn` (if 5D tensor) as well as `.fftshift` and `.ifftshift`
    to shift the low frequencies from the outer corner into the center of the k-space.

    Args:
        scan (torch.Tensor): Complex valued 2D (H X W), 3D (C X H X W), 4D (B x C x H x W) or 5D (B x C x H x W x D)
            input tensor in k-space.
        dim (tuple[int, int]): Dimension over which to perform the transformation.

    Returns:
        torch.Tensor: Complex valued inverse fourier transformed 4D or 5D tensor in image space.
    """
    if scan.ndim not in [2, 3, 4, 5]:
        raise ValueError(
            f"Dimension of input needs to be 2, 3, 4 or 5 (B x C x H x W x D), but got {scan.ndim}!"
        )
    if scan.ndim == 2:
        scan = torch.fft.ifftshift(scan)
        scan = torch.fft.ifft2(scan)
        scan = torch.fft.fftshift(scan)
    elif scan.ndim == 3:
        scan = torch.fft.ifftshift(scan, dim=dim)
        scan = torch.fft.ifft2(scan, dim=dim)
        scan = torch.fft.fftshift(scan, dim=dim)
    elif scan.ndim == 4:
        scan = torch.fft.ifftshift(scan, dim=dim)
        scan = torch.fft.ifftn(scan, dim=dim)
        scan = torch.fft.fftshift(scan, dim=dim)
    elif scan.ndim == 5:
        scan = torch.fft.ifftshift(scan, dim=dim)
        scan = torch.fft.ifftn(scan, dim=dim)
        scan = torch.fft.fftshift(scan, dim=dim)

    return scan


@torch.jit.script
def fft(scan: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:
    """Calculate the fourier transform of an complex valued input tensor
    in the image domain to transform it into the k-space (frequency domain).

    Use the fast fourier transform (FFT) module `torch.fft.fft2` (if 4D tensor)
    or `torch.fft.fftn` (if 5D tensor) as well as `.fftshift` and `.ifftshift`
    to shift the low frequencies from the outer corner into the center of the k-space.

    Args:
        scan (torch.Tensor): Complex valued 2D (H X W), 3D (C X H X W), 4D (B x C x H x W) or 5D (B x C x H x W x D)
            input tensor in image space.
        dim (tuple[int, int]): Dimension over which to perform the transformation.

    Returns:
        torch.Tensor: Complex valued inverse fourier transformed 4D or 5D tensor in k-space.
    """
    if scan.ndim not in [2, 3, 4, 5]:
        raise ValueError(
            f"Dimension of input needs to be 2, 3, 4 or 5 (B x C x H x W x D), but got {scan.ndim}!"
        )

    if scan.ndim == 2:
        scan = torch.fft.ifftshift(scan)
        scan = torch.fft.fft2(scan)
        scan = torch.fft.fftshift(scan)
    elif scan.ndim == 3:
        scan = torch.fft.ifftshift(scan, dim=dim)
        scan = torch.fft.fft2(scan, dim=dim)
        scan = torch.fft.fftshift(scan, dim=dim)
    elif scan.ndim == 4:
        scan = torch.fft.ifftshift(scan, dim=dim)
        scan = torch.fft.fftn(scan, dim=dim)
        scan = torch.fft.fftshift(scan, dim=dim)
    elif scan.ndim == 5:
        scan = torch.fft.ifftshift(scan, dim=dim)
        scan = torch.fft.fftn(scan, dim=dim)
        scan = torch.fft.fftshift(scan, dim=dim)

    return scan
