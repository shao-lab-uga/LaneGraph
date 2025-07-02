import torch
import torch.nn as nn
import torchvision.models as models


class Resnet34Classifier(nn.Module):
    def __init__(self, num_image_classes=2, pretrained=False):
        super().__init__()
        self.model = models.resnet34(pretrained=pretrained)

        # Replace the original 1000-class FC layer with a custom one
        self.model.fc = nn.Linear(512, num_image_classes)

    def forward(self, x):
        return self.model(x)