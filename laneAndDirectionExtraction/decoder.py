import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=64):
        super().__init__()
        encoder_channels = encoder_channels[::-1]

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1) for ch in encoder_channels
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in encoder_channels
        ])

    def forward(self, features):
        features = features[::-1]
        lateral_features = [l_conv(f) for l_conv, f in zip(self.lateral_convs, features)]

        x = lateral_features[0]
        for i in range(1, len(lateral_features)):
            x = F.interpolate(x, size=lateral_features[i].shape[2:], mode='bilinear', align_corners=False)
            x = x + lateral_features[i]
            x = self.smooth_convs[i](x)

        # Final upsample to match 640×640
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ViTUNetDecoder(nn.Module):
    def __init__(self, in_channels, base_channels=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.up1 = UpConvBlock(base_channels, base_channels // 2)   # 40 → 80
        self.up2 = UpConvBlock(base_channels // 2, base_channels // 4)  # 80 → 160
        self.up3 = UpConvBlock(base_channels // 4, base_channels // 8)  # 160 → 320
        self.up4 = UpConvBlock(base_channels // 8, base_channels // 16) # 320 → 640

        self.out_channels = base_channels // 16  # for final_conv layer

    def forward(self, features):
        x = features[0]  # ViT feature map, shape [B, C, 40, 40]
        x = self.proj(x)        # → [B, 256, 40, 40]
        x = self.up1(x)         # → [B, 128, 80, 80]
        x = self.up2(x)         # → [B, 64, 160, 160]
        x = self.up3(x)         # → [B, 32, 320, 320]
        x = self.up4(x)         # → [B, 16, 640, 640]
        return x