# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torchvision
from torch import nn, Tensor


class VGGLoss(nn.Module):
    """
    Calculate loss based on the official pre-trained VGG19 model from
    torchvision.

    Parameters
    ----------
    feature_layer : int
        An ``int`` of the number of features to pull from the model.
    """
    def __init__(self, feature_layer: int = 36) -> None:
        super(VGGLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.features.children())[:feature_layer]).eval()

        for _, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, source: Tensor, target: Tensor) -> float:
        """
        Calculate the loss using the pre-trained VGG19 model.

        Parameters
        ----------
        source : Tensor
            A ``tensor`` of the generated upscaled image.
        target : Tensor
            A ``tensor`` of the high resolution source image.

        Returns
        -------
        float
            Returns a ``float`` of the calculated loss.
        """
        loss = torch.nn.functional.l1_loss(self.features(source),
                                           self.features(target))
        return loss
