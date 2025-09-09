import torch
import torchvision
import torch.nn as nn

class DecoderBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        conv_in_channels,
        conv_out_channels,
        up_in_channels=None,
        up_out_channels=None,
    ):
        super().__init__()
        """
        eg:
        decoder1:
        up_in_channels      : 1024,     up_out_channels     : 512
        conv_in_channels    : 1024,     conv_out_channels   : 512

        decoder5:
        up_in_channels      : 64,       up_out_channels     : 64
        conv_in_channels    : 128,      conv_out_channels   : 64
        """
        if up_in_channels == None:
            up_in_channels = conv_in_channels
        if up_out_channels == None:
            up_out_channels = conv_out_channels

        self.up = nn.ConvTranspose2d(
            up_in_channels, up_out_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                conv_in_channels,
                conv_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                conv_out_channels,
                conv_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
        )

    # x1-upconv , x2-downconv
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UnetResnet34(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.in_channels = model_config.in_channels  # new input channels

        resnet34 = torchvision.models.resnet34(weights='DEFAULT')  # Use pretrained
        filters = [64, 128, 256, 512]

        # Replace the first conv layer to support arbitrary input channels
        resnet34.conv1 = nn.Conv2d(
            self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.firstlayer = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
        )
        self.maxpool = resnet34.maxpool
        self.encoder1 = resnet34.layer1
        self.encoder2 = resnet34.layer2
        self.encoder3 = resnet34.layer3
        self.encoder4 = resnet34.layer4

        self.bridge = nn.Sequential(
            nn.Conv2d(filters[3], filters[3] * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[3] * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder1 = DecoderBlock(
            conv_in_channels=filters[3] * 2, conv_out_channels=filters[3]
        )
        self.decoder2 = DecoderBlock(
            conv_in_channels=filters[3], conv_out_channels=filters[2]
        )
        self.decoder3 = DecoderBlock(
            conv_in_channels=filters[2], conv_out_channels=filters[1]
        )
        self.decoder4 = DecoderBlock(
            conv_in_channels=filters[1], conv_out_channels=filters[0]
        )
        self.decoder5 = DecoderBlock(
            conv_in_channels=filters[1],
            conv_out_channels=filters[0],
            up_in_channels=filters[0],
            up_out_channels=filters[0],
        )

        self.lastlayer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=filters[0], out_channels=filters[0], kernel_size=2, stride=2
            ),
            nn.Conv2d(filters[0], 4, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        e1 = self.firstlayer(x)      # (B, ch_in, H, W) -> (B, 64, H/2, W/2)
        maxe1 = self.maxpool(e1)     # -> (B, 64, H/4, W/4)
        e2 = self.encoder1(maxe1)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        e5 = self.encoder4(e4)

        c = self.bridge(e5)

        d1 = self.decoder1(c, e5)
        d2 = self.decoder2(d1, e4)
        d3 = self.decoder3(d2, e3)
        d4 = self.decoder4(d3, e2)
        d5 = self.decoder5(d4, e1)

        out = self.lastlayer(d5)
        lane_map = out[:, :2, :, :]
        direction_map = out[:, 2:, :, :]
        return lane_map, direction_map