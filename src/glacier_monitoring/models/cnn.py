"""Residual U-Net for glacier segmentation, trained from scratch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-layer ResNet basic block with residual skip connection and spatial dropout."""

    def __init__(self, in_c: int, out_c: int, dropout_p: float = 0.15) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout2d(p=dropout_p)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
            if in_c != out_c
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class ResUNet(nn.Module):
    """Small U-Net with residual blocks: three down-sampling stages and skip connections.

    Input height and width must be divisible by 8.
    """

    def __init__(
        self,
        in_channels: int = 14,
        num_classes: int = 2,
        base_channels: int = 24,
        dropout_p: float = 0.15,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 6

        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False)
        self.enc1 = ResidualBlock(c1, c1, dropout_p=dropout_p * 0.5)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(c1, c2, dropout_p=dropout_p)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(c2, c3, dropout_p=dropout_p)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(c3, c4, dropout_p=dropout_p)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(c3 + c3, c3, dropout_p=dropout_p)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(c2 + c2, c2, dropout_p=dropout_p)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(c1 + c1, c1, dropout_p=dropout_p * 0.5)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = F.gelu(self.stem(x))
        e1 = self.enc1(s0)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Glacier probability map of shape ``(B, H, W)``, returned on the CPU."""
        self.eval()
        with torch.no_grad():
            x = x.to(next(self.parameters()).device)
            return F.softmax(self.forward(x), dim=1)[:, 1].cpu()
