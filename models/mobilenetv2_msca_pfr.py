import torch
import torch.nn as nn
from torchvision import models

from .msca import MSCA
from .pfr import PoolFormerBlock, GroupNorm


class MobileNetV2_MSCA_PFR(nn.Module):
    """
    MobileNetV2 backbone + MSCA (ASPP) + PFR block + classifier.
    EXACT behavior preserved from your original implementation.
    """

    def __init__(self, num_classes=4):
        super().__init__()

        # Load pretrained MobileNetV2
        backbone = models.mobilenet_v2(pretrained=True)

        # Feature extractor
        self.features = backbone.features  # output: 1280 channels

        # Reduce MobileNetV2's high dimension → 256
        self.reduce = nn.Conv2d(1280, 256, kernel_size=1)

        # Multi-Scale Context Aggregator
        self.msca = MSCA(in_channels=256, out_channels=256)

        # Pool-based Feature Refinement block
        self.pfr = PoolFormerBlock(
            dim=256,
            pool_size=3,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=GroupNorm,
            drop=0.1
        )

        # Skip connection to match original behavior
        self.skip = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # MobileNetV2 features
        x = self.features(x)

        # Reduce dimension
        x = self.reduce(x)

        # MSCA block
        x = self.msca(x)

        # Skip connection
        skip = self.skip(x)

        # PFR block + skip
        x = self.pfr(x) + skip

        # Global average pooling
        x = x.mean([2, 3])

        # Classification
        x = self.classifier(x)

        return x
