import torch.nn as nn
import torchvision


class Resnet34Classifier(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.num_classes = model_config.num_classes
        self.in_channels = model_config.in_channels
        self.model = torchvision.models.resnet34(weights='DEFAULT')  # Use pretrained weights
        
        self.model.conv1 = nn.Conv2d(
            self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Replace the original 1000-class FC layer with a custom one
        self.model.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        return self.model(x)