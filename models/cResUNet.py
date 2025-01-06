# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import torch.nn as nn
import models.layer.spectral_blocks as layer
import models.layer.spectral_layer as parts
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Literal


class ResUNet(nn.Module):
    """Implementation of a complex valued Residual U-Net model.

    Args:
        config (dict): Configuration dictionary.
        features (list): List of feature channels for each layer.
        device (torch.device): Device for computation.
        activation (nn.Module): Activation function for the model.
        padding (int, optional): Padding size. Defaults to None.
        dilation (int, optional): Dilation rate for convolutional layers. Defaults to 1.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        logger: Logger for debugging. Defaults to None.
        viz (bool, optional): Flag for visualization. Defaults to False.

    Attributes:
        downs (nn.ModuleList): List of downsampling layers.
        bottleneck (nn.ModuleList): List of bottleneck layers.
        ups (nn.ModuleList): List of upsampling layers.
        pool: Spectral pooling layer.
        final_layer: Final convolutional layer.

    Methods:
        forward(x): Forward pass of the ResUNet model.

    Note:
        The model is designed for complex-valued inputs.
    """

    def __init__(
        self,
        config: dict,
        features: list,
        device: torch.device | list[torch.device],
        activation: nn.Module,
        padding: int = None,
        dilation: int = 1,
        in_channels: int = 1,
        out_channels: int = 1,
        logger=None,
        viz=False,
    ) -> None:
        super(ResUNet, self).__init__()

        self.logger = logger
        self.downs = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pooling_size = config["pooling_size"]
        self.pool = parts.SpectralPool(kernel_size=self.pooling_size)
        self.padding = padding
        self.kernel_size = config["kernel_size"]

        # Set default padding if not provided
        if padding == None:
            self.padding = int((self.kernel_size - 1) / 2)

        self.device = device
        self.dropout = config["dropout"]
        self.dilation = dilation
        self.activation = activation
        self.viz = viz
        self.i = 0
        Conv = nn.Conv2d
        #Conv = parts.FFTConv
        self.res_length = config["length"]

        # Create downsampling blocks
        for feature in features:
            self.downs.append(
                layer.ResidualBlock(
                    in_channels=in_channels,
                    out_channels=feature,
                    dropout=self.dropout,
                    kernel_size=self.kernel_size,
                    device=self.device,
                    padding=self.padding,
                    stride=1,
                    dilation=self.dilation,
                    activation=self.activation,
                    resample=True,
                )
            )
            in_channels = feature
            for i in range(self.res_length):
                self.downs.append(
                    layer.ResidualBlock(
                        in_channels=feature,
                        out_channels=feature,
                        dropout=self.dropout,
                        kernel_size=self.kernel_size,
                        device=self.device,
                        padding=self.padding,
                        stride=1,
                        dilation=self.dilation,
                        activation=self.activation,
                        resample=False,
                    )
                )

        # Create bottleneck block
        bottleneck_in_channels = features[-1]
        bottleneck_out_channels = features[-1] * 2
        self.bottleneck.append(
            layer.ResidualBlock(
                in_channels=bottleneck_in_channels,
                out_channels=bottleneck_out_channels,
                dropout=self.dropout,
                kernel_size=self.kernel_size,
                device=self.device,
                padding=self.padding,
                stride=1,
                dilation=self.dilation,
                activation=self.activation,
                resample=True,
            )
        )

        # Create upsampling block
        for feature in reversed(features):
            self.ups.append(
                layer.Upsampling(
                    in_channels=feature * 2,
                    out_channels=feature,
                    device=self.device,
                    scale_factor=self.pooling_size,
                    padding=self.padding,
                    kernel_size=self.kernel_size,
                )
            )
            self.ups.append(
                layer.ResidualBlock(
                    in_channels=feature * 2,
                    out_channels=feature,
                    dropout=0,
                    kernel_size=self.kernel_size,
                    device=self.device,
                    padding=self.padding,
                    stride=1,
                    dilation=self.dilation,
                    activation=self.activation,
                    resample=True,
                )
            )

        # Create final layer
        self.final_layer = Conv(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            device=self.device,
            dtype=torch.complex64,
        )

    def forward(self, x):
        self.viz = False

        if self.logger is not None:
            if self.logger.level == 10:
                self.i += 1
                if self.i % 1000 == 0:
                    ResUNet.__layer_debugging(self)

        skip_connections = []
        cut_offs = []

        # Downsample blocks
        for idx, down in enumerate(self.downs):
            x = down(x)
            if self.viz is True:
                ResUNet.__filter_visualization(x, idx, "down")

            # Save intermediate outputs after downsampling
            if idx % (self.res_length + 1) == 0:
                skip_connections.append(x)
                x, cut_off = self.pool(x)
                cut_offs.append(cut_off)

        # Bottleneck block
        for bottleneck in self.bottleneck:
            x = bottleneck(x)
        skip_connections = skip_connections[::-1]
        cut_offs = cut_offs[::-1]

        # Upsample blocks
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x, cut_offs[idx // 2])
            skip_connection = skip_connections[idx // 2]
            if self.viz is True:
                ResUNet.__filter_visualization(x, idx, "up")

            # Ensure output size matches skip_connection output size
            if x.shape != skip_connection.shape:
                x_real = nn.Upsample(size=skip_connection.shape[2:])(x.real)
                x_imag = nn.Upsample(size=skip_connection.shape[2:])(x.imag)
                x = x_real + 1j * x_imag

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)

        # Final layer
        x = self.final_layer(x)

        if x.ndim == 5:
            x = x.permute(0, 1, 3, 4, 2)

        return x

    def __layer_debugging(self) -> None:
        """
        Debugging method to log information about model parameters during training.

        This method logs the maximum and minimum values of parameters and gradients for each layer in the model.
        It is designed to be called periodically during training to monitor the behavior of the model.

        Returns:
            None
        """

        self.logger.debug(
            f"Iteration {self.i}\r_________________________________________________________________________"
        )

        # Loop through all named parameters in the model
        for name, param in self.named_parameters():
            try:
                # Log information about parameter values and gradients
                self.logger.debug(
                    f"\r\
                {name} Max {param.max().item()} | Min {param.min().item()} \r\
                {name}.grad Max {param.grad.max().item()} | Min {param.grad.min().item()}"
                )
            except:
                try:
                    # Log information about absolute parameter values and gradients
                    self.logger.debug(
                        f"\r\
                    {name} Max {param.abs().max().item()} | Min {param.abs().min().item()} \r\
                    {name}.grad Max {param.grad.abs().max().item()} | Min {param.grad.abs().min().item()}"
                    )
                except:
                    continue

    def __filter_visualization(x, idx: int, part: str = Literal["down", "up"]):
        """
        Visualization method to create and save plots of convolutional filters.

        This method generates a visual representation of convolutional filters from the input tensor `x`.
        It creates a 3x3 grid of subplots, each showing the absolute values of filters from the first batch.
        The visualization is saved as an image file in the specified directory.

        Args:
            x (torch.Tensor): The input tensor containing convolutional filters.
            idx (int): The index of the convolutional layer for identification in the saved plot.
            part (str): A string indicating whether the filters are from the "down" or "up" part of the model.

        Returns:
            None
        """

        plt.rcParams["figure.dpi"] = 500
        plt.rcParams["font.size"] = 17
        plt.rcParams.update({"figure.figsize": (10, 10)})
        plt.figure(figsize=(10, 10))

        # Iterate through the filters in the first batch
        for i, filter in enumerate(x[0, ...]):
            print(i)
            if i == 9:  # we will visualize only 4x4 blocks from each layer
                break

            plt.subplot(3, 3, i + 1)
            im = plt.imshow(filter.abs().detach().cpu(), cmap="gray")
            plt.imshow(filter.abs().detach().cpu(), cmap="gray")
            plt.axis("off")

        cax = plt.axes([0.88, 0.1, 0.01, 0.75])
        plt.colorbar(im, cax=cax)
        plt.suptitle(f"{part}Conv {idx}, Input size {list(x.size())}", x=0.45, y=0.95)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(Path.home() / f"debugging_plots/{part}layer{idx}.png")
        plt.close()
