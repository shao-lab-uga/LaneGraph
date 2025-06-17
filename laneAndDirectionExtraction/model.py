from torch import nn
from laneAndDirectionExtraction.backbone import BackboneWrapper, ViTBackboneWrapper
from laneAndDirectionExtraction.decoder import FPNDecoder, ViTUNetDecoder
import torch
class LaneAndDirectionExtractionModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.backbone_name = model_config.backbone_name
        self.decoder_type = model_config.decoder_type
        self.output_dim = model_config.output_dim
        if any(vit_type in self.backbone_name.lower() for vit_type in ['vit', 'coat', 'maxvit']):
            self.encoder = ViTBackboneWrapper(self.backbone_name)
        else:
            self.encoder = BackboneWrapper(self.backbone_name)

        channels = self.encoder.out_channels
        for channel in channels:
            print(f"Encoder channel: {channel}")
        if self.decoder_type == 'fpn':
            self.decoder = FPNDecoder(channels, out_channels=64)
            final_channels = 64
        elif self.decoder_type == 'vitdecoder':
            self.decoder = ViTUNetDecoder(in_channels=channels[0], base_channels=256)
            final_channels = 256 // (2 ** (4-1)) 
        else:
            raise ValueError(f"Unsupported decoder_type: {self.decoder_type}")

        self.final_conv = nn.Conv2d(final_channels, self.output_dim, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        x = self.decoder(feats)
        x = self.final_conv(x)
        return x.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]

import torch
from laneAndDirectionExtraction.model import LaneAndDirectionExtractionModel

def test_model(backbone_name, decoder_type, output_dim=3, input_size=640):
    print(f"🧪 Testing: Backbone = {backbone_name}, Decoder = {decoder_type}")

    model = LaneAndDirectionExtractionModel(
        backbone_name=backbone_name,
        decoder_type=decoder_type,
        output_dim=output_dim
    )
    model.eval()

    x = torch.randn(2, input_size, input_size, 3)  # [B, H, W, 3]
    with torch.no_grad():
        y = model(x)

    print(f"✅ Output shape: {y.shape}\n")
    assert y.shape[1] == input_size and y.shape[2] == input_size, "Output shape mismatch!"

# ====================
# Define test settings
# ====================
test_settings = [
    # ("resnet34", "fpn"),
    # ("resnet50", "fpn"),
    ('vit_base_patch16_224.dino', 'vitdecoder'),
]

if __name__ == "__main__":
    for backbone, decoder in test_settings:
        test_model(backbone, decoder)
