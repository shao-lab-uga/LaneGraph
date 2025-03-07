import torch
import torch.nn as nn
import torchvision


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
        if up_in_channels is None:
            up_in_channels = conv_in_channels
        if up_out_channels is None:
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
    def __init__(self, ch_in, ch_out):
        super().__init__()
        resnet34 = torchvision.models.resnet34(weights=None)
        filters = [64, 128, 256, 512]
        self.firstlayer = nn.Sequential(
            nn.Conv2d(ch_in, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # self.firstlayer = nn.Sequential(*list(resnet34.children())[:3])
        self.maxpool = list(resnet34.children())[3]
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
            nn.Conv2d(filters[0], ch_out, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        e1 = self.firstlayer(x)  # N,ch_in,640,640->N,64,320,320
        maxe1 = self.maxpool(e1)  # N,64,320,320->N,64,160,160
        e2 = self.encoder1(maxe1)  # N,64,160,160->N,64,160,160
        e3 = self.encoder2(e2)  # N,64,160,160->N,128,80,80
        e4 = self.encoder3(e3)  # N,128,80,80->N,256,40,40
        e5 = self.encoder4(e4)  # N,256,40,40->N,512,   20,20

        c = self.bridge(e5)  # N,512,20,20->N,1024,10,10

        d1 = self.decoder1(c, e5)  # N,512,20,20
        d2 = self.decoder2(d1, e4)  # N,256,40,40
        d3 = self.decoder3(d2, e3)  # N,128,80,80
        d4 = self.decoder4(d3, e2)  # N,64,160,160
        d5 = self.decoder5(d4, e1)  # N,64,320,320

        out = self.lastlayer(d5)  # N,ch_out,640,640

        return out


class LazyResnet18Classifier(nn.Module):
    """
    Qucikly made resnet18 classifier based on torchvision model.
    Does not match Songtaohe version. Only changed to support dynamic chanels in and out.
    """

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        # changing first layer for ch_in
        self.model.conv1 = nn.Conv2d(
            ch_in, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # changing last layer for ch_out
        self.model.fc = nn.Linear(in_features=512, out_features=ch_out)

    def forward(self, x):
        out = self.model(x)
        return out
