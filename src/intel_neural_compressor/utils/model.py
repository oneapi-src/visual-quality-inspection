# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""System module."""
# pylint: disable=E1101,E1102,E0401,R0801
from torch import nn
from torchvision import models
from utils.constants import INPUT_IMG_SIZE


class CustomVGG(nn.Module):
    """
    Custom multi-class classification model
    with VGG16 feature extractor, pretrained on ImageNet
    and custom classification head.
    Parameters for the first convolutional blocks are freezed.
    Returns class scores when in train mode.
    Returns class probs and normalized feature maps when in eval mode.
    """

    def __init__(self, n_classes=2):
        super().__init__()
        self.feature_extractor = models.vgg16(pretrained=True).features[:-1]
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(
                kernel_size=(INPUT_IMG_SIZE[0] // 2 ** 5, INPUT_IMG_SIZE[1] // 2 ** 5)
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=self.feature_extractor[-2].out_channels,
                out_features=n_classes,
            ),
        )
        # self._freeze_params()

    def _freeze_params(self):
        for param in self.feature_extractor[:23].parameters():
            param.requires_grad = False

    def forward(self, x_in):
        """
        forward
        """
        feature_maps = self.feature_extractor(x_in)
        scores = self.classification_head(feature_maps)

        if self.training:
            return scores

        probs = nn.functional.softmax(scores, dim=-1)
        return probs
