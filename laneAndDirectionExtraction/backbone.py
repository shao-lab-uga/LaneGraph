import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class BackboneWrapper(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, pretrained=True)
        self.out_channels = self.out_channels = [info['num_chs'] for info in self.backbone.feature_info]

    def forward(self, x):
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # [B, H, W, 3] → [B, 3, H, W]
        features = self.backbone(x)
        return features
    

class ViTBackboneWrapper(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.model = timm.create_model(backbone_name, 
                                       pretrained=True,
                                       dynamic_img_size=True,
                                       dynamic_img_pad=True)

        self.has_features_only = hasattr(self.model, 'feature_info')

        if self.has_features_only:
            self.model = timm.create_model(backbone_name, 
                                           features_only=True, 
                                           pretrained=True,
                                           dynamic_img_size=True,
                                           dynamic_img_pad=True)
            self.forward_method = self._forward_cnn_style
            # FIX: extract channels from feature_info manually
            self.out_channels = [info['num_chs'] for info in self.model.feature_info][::-1]
        else:
            self.forward_method = self._forward_vit_style
            self.out_channels = [self.model.num_features]

    def _forward_cnn_style(self, x):
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        return self.model(x)[::-1]

    def _forward_vit_style(self, x):
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        f = self.model.forward_features(x)
        return [f]

    def forward(self, x):
        return self.forward_method(x)