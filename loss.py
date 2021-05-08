import torch
import torchvision
from torch import nn


class VGGLoss(nn.Module):
    def __init__(self, feature_layer=36):
        super(VGGLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.features.children())[:feature_layer]).eval()

        for name, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, source, target):
        loss = torch.nn.functional.l1_loss(self.features(source), self.features(target))
