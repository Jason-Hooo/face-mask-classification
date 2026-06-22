
import torch.nn as nn
from math import ceil

mb_params = [
    # expand_ratio, channels, repeated_times, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

alpha, beta = 1.2, 1.1

scale_values = {
    # (Φ=phi, resolution, dropout_rate)
    "B0": (0, 224, 0.2),
    "B1": (0.5, 240, 0.2),
    "B2": (1, 260, 0.3),
    "B3": (2, 300, 0.3),
    "B4": (3, 380, 0.4),
    "B5": (4, 456, 0.4),
    "B6": (5, 528, 0.5),
    "B7": (6, 600, 0.5),
}

class ConvBnSiLUBlock(nn.Module):
    """
    Standard convolution block: Conv2d -> BatchNorm2d -> SiLU.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Convolution kernel size.
        stride (int): Convolution stride.
        padding (int): Padding size.
        groups (int, optional): Depthwise group size. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBnSiLUBlock, self).__init__()
        self.block = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups), # groups: 分成幾組做Depthwise Convolution，預設是1（不做）
                        nn.BatchNorm2d(out_channels), 
                        nn.SiLU()
                    )

    def forward(self, x):
        return self.block(x)
    
class SEBlock(nn.Module): 
    """
    Squeeze-and-Excitation (SE) block for channel-wise feature recalibration.

    Args:
        in_channels (int): Input channels.
        reduced_channels (int): Squeezed intermediate channels.
    """
    def __init__(self, in_channels, reduced_channels):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_channels, reduced_channels, 1), 
            nn.SiLU(), 
            nn.Conv2d(reduced_channels, in_channels, 1), 
            nn.Sigmoid(), 
        )

    def forward(self, x):
        return x * self.se(x) 
    
class MBConvBlock(nn.Module): 
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block with optional residual connection.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Depthwise convolution kernel size.
        stride (int): Convolution stride.
        padding (int): Padding size.
        expand_ratio (int): Channel expansion multiplier.
        reduction (int, optional): SE block reduction ratio. Defaults to 4.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=2): 
        super(MBConvBlock, self).__init__()

        expanded_channels = in_channels * expand_ratio
        self.is_expand = expand_ratio > 1
        reduced_channels = max(1, int(in_channels / reduction)) 

        if self.is_expand:
            self.expand_conv = ConvBnSiLUBlock(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(
            ConvBnSiLUBlock(expanded_channels, expanded_channels, kernel_size, stride, padding, groups=expanded_channels),
            SEBlock(expanded_channels, reduced_channels),
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.is_expand:
            return self.conv(self.expand_conv(x))
        else:
            return self.conv(x)
        
class EfficientNet(nn.Module):
    """
    EfficientNet architecture dynamically scaled by B0-B7 parameters.

    Args:
        model_name (str): EfficientNet variant (e.g., 'B0').
        out_classes (int): Number of classification output classes.
    """
    def __init__(self, model_name, out_classes):
        super(EfficientNet, self).__init__()
        phi, resolution, dropout_rate = scale_values[model_name]
        self.depth_multiplier, self.width_multiplier = alpha**phi, beta**phi
        self.last_channels = ceil(1280 * self.width_multiplier) 
        self.blocks_construction()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.last_channels, out_classes)
        )

    def blocks_construction(self):
        """Constructs the sequence of MBConv blocks based on scaling multipliers."""
        channels = int(32 * self.width_multiplier)
        self.blocks = [ConvBnSiLUBlock(in_channels=3, out_channels=channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels
        for expand_ratio, channels, repeated_times, stride, kernel_size in mb_params:
            out_channels = 4 * ceil(int(channels * self.width_multiplier) / 4)
            num_layers = ceil(repeated_times * self.depth_multiplier)

            for layer in range(num_layers):
                if layer != 0:
                    stride = 1
                self.blocks.append(
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                )
                in_channels = out_channels
        self.blocks.append(ConvBnSiLUBlock(in_channels, self.last_channels, kernel_size=1, stride=1, padding=0))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, out_classes).
        """
        avgpool= nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()
        return self.classifier(flatten(avgpool(self.blocks(x))))