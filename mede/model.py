# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import timm
from mede.mednext.MedNextV1 import MedNeXt


class ConvNext(nn.Module):
    """
    ConvNext is a PyTorch module that implements a convolutional neural network for segmentation tasks.
    It consists of an encoder and a decoder, with skip connections between them.

    Args:
        num_decoder_blocks (int): The number of decoder blocks in the network.

    Attributes:
        encoder (nn.Module): The encoder module, which is a pretrained ConvNext model from the timm library.
        decoder (nn.ModuleList): The decoder module, which is a list of decoder blocks.
        final (nn.Sequential): The segmentation head module, which performs the final convolution.

    Methods:
        decoder_block: Creates a decoder block module.
        segmentation_head: Creates a segmentation head module.
        forward: Performs a forward pass through the network.

    """

    def __init__(self, num_decoder_blocks: int = 4) -> None:
        super(ConvNext, self).__init__()
        self._num_decoder_blocks = num_decoder_blocks
        self.dim = 3
        self.encoder = timm.create_model(
            "convnext_atto", pretrained=True, features_only=True, in_chans=1
        )
        self.decoder = nn.ModuleList()
        in_channels = 640
        out_channels = in_channels // 4

        for i in range(self._num_decoder_blocks):
            self.decoder.append(self.decoder_block(in_channels, out_channels))
            in_channels = out_channels * 2  # for skip connections
            out_channels = out_channels // 2
        self.decoder.append(self.decoder_block(out_channels * 2, out_channels))
        self.final = self.segmentation_head(out_channels, 1)

    @staticmethod
    def decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Creates a decoder block module.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        Returns:
            nn.Sequential: The decoder block module.

        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    @staticmethod
    def segmentation_head(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Creates a segmentation head module.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        Returns:
            nn.Sequential: The segmentation head module.

        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
    
    def _forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x_in (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        feature_maps = self.encoder(x_in)[::-1]
        out = feature_maps[0]
        for idx, layer in enumerate(self.decoder):
            if idx in range(len(feature_maps)):
                out = torch.cat([out, feature_maps[idx]], dim=1)
            out = layer(out)
        out = self.final(out)
        return out

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if x_in.ndim == 5:
            for i in range(x_in.size(2)):
                x = x_in[:,:, i, ...]
                x = self._forward(x)
                if i == 0:
                    out = x.unsqueeze(2)
                else:
                    out = torch.cat((out, x.unsqueeze(2)), 2)
            return out
        else:
            return self._forward(x_in)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(UNet3D, self).__init__()

        # Encoder (contracting path)
        self.encoder_conv1 = self.conv_block(in_channels, 64)
        self.encoder_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder_conv2 = self.conv_block(64, 128)
        self.encoder_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder_conv3 = self.conv_block(128, 256)
        self.encoder_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = self.conv_block(256, 512)

        # Decoder (expansive path)
        self.decoder_upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv3 = self.conv_block(512, 256)
        self.decoder_upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv2 = self.conv_block(256, 128)
        self.decoder_upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv1 = self.conv_block(128, 64)

        # Output layer
        self.output_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.encoder_conv1(x)
        enc_pool1 = self.encoder_pool1(enc1)
        enc2 = self.encoder_conv2(enc_pool1)
        enc_pool2 = self.encoder_pool2(enc2)
        enc3 = self.encoder_conv3(enc_pool2)
        enc_pool3 = self.encoder_pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck_conv(enc_pool3)

        # Decoder
        dec_upconv3 = self.decoder_upconv3(bottleneck)
        dec_concat3 = torch.cat([dec_upconv3, enc3], dim=1)
        dec_conv3 = self.decoder_conv3(dec_concat3)
        dec_upconv2 = self.decoder_upconv2(dec_conv3)
        dec_concat2 = torch.cat([dec_upconv2, enc2], dim=1)
        dec_conv2 = self.decoder_conv2(dec_concat2)
        dec_upconv1 = self.decoder_upconv1(dec_conv2)
        dec_concat1 = torch.cat([dec_upconv1, enc1], dim=1)
        dec_conv1 = self.decoder_conv1(dec_concat1)

        # Output layer
        output = self.output_conv(dec_conv1)

        return output

class Mednext(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(Mednext, self).__init__()

        self.model = MedNeXt(
            in_channels = in_channels, 
            n_channels = 32,
            n_classes = out_channels, 
            exp_r=2,                         
            kernel_size=3,         
            deep_supervision=False,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

    def forward(self, x):

        return self.model(x)
