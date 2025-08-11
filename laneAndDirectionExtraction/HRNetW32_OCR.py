import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ----------------------------
# OCR modules (per Wang et al., CVPR'20)
# ----------------------------

class SpatialGatherModule(nn.Module):
    """
    Aggregate the context features according to the predicted soft object regions (probs).
    feats:  [B, C, H, W]
    probs:  [B, K, H, W]  (soft region probabilities; we can synthesize K=1 to build global context)
    return: [B, C, K, 1]  context vectors
    """
    def __init__(self):
        super().__init__()

    def forward(self, feats, probs):
        B, C, H, W = feats.size()
        _, K, _, _ = probs.size()
        feats = feats.view(B, C, -1)                  # [B, C, HW]
        probs = probs.view(B, K, -1)                  # [B, K, HW]
        probs = F.softmax(probs, dim=2)               # across spatial
        context = torch.bmm(feats, probs.transpose(1, 2))  # [B, C, K]
        context = context.unsqueeze(-1)               # [B, C, K, 1]
        return context


class ObjectAttentionBlock2D(nn.Module):
    """
    Object context attention (similar to non-local but keyed by object context).
    x: [B, C, H, W]
    proxy: [B, C, K, 1]
    """
    def __init__(self, in_channels, key_channels, value_channels, scale=1):
        super().__init__()
        self.scale = scale
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key   = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.f_query = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels, value_channels, kernel_size=1)
        self.f_out   = nn.Conv2d(value_channels, in_channels, kernel_size=1)

    def forward(self, x, proxy):
        B, C, H, W = x.size()
        K = proxy.size(2)

        query = self.f_query(x).view(B, self.key_channels, -1)              # [B, Kq, HW]
        query = query.permute(0, 2, 1)                                      # [B, HW, Kq]

        key = self.f_key(proxy.squeeze(-1))                                 # [B, Kc, K]
        key = key.permute(0, 2, 1)                                          # [B, K, Kc]

        value = self.f_value(proxy.squeeze(-1))                              # [B, Vc, K]
        value = value.permute(0, 2, 1)                                      # [B, K, Vc]

        sim_map = torch.bmm(query, key)                                     # [B, HW, K]
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.bmm(sim_map, value)                                  # [B, HW, Vc]
        context = context.permute(0, 2, 1).contiguous().view(B, self.value_channels, H, W)
        out = self.f_out(context)
        return out


class SpatialOCRModule(nn.Module):
    """
    Spatial OCR: gather object context then distribute back via attention and fuse.
    """
    def __init__(self, in_channels, key_channels=64, value_channels=256, out_channels=512, num_regions=1):
        super().__init__()
        self.num_regions = num_regions
        self.gather = SpatialGatherModule()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, value_channels)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # A tiny region predictor to form soft object regions (K maps); here K=num_regions (often 1 for global)
        self.region_pred = nn.Conv2d(in_channels, num_regions, kernel_size=1)

    def forward(self, feats):
        # probs from region predictor
        probs = self.region_pred(feats)                        # [B, K, H, W]
        context = self.gather(feats, probs)                    # [B, C, K, 1]
        obj_attended = self.object_context_block(feats, context)   # [B, C, H, W]
        out = torch.cat([obj_attended, feats], dim=1)          # concat context-enhanced + original
        out = self.conv_bn_relu(out)                           # [B, out_channels, H, W]
        return out


# ----------------------------
# HRNetW32 + OCR + 3 heads
# ----------------------------

class HRNetW32_OCR(nn.Module):
    """
    Input:  [B, 3, H, W] (e.g., 640x640)
    Outputs:
      - lane_mask_logits: [B, 1, H, W]
      - dir_map:         [B, 2, H, W] (unit vectors; normalized in forward)
      - angle_bin_logits:[B, K, H, W]
    """
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.ocr_key_ch = model_config.ocr_key
        self.ocr_val_ch = model_config.ocr_val
        self.ocr_out_ch = model_config.ocr_out
        self.context_regions = model_config.context_regions
        # HRNetV2-W32 backbone (timm). Returns 4 multi-resolution features at strides [4, 8, 16, 32]
        self.backbone = timm.create_model('hrnet_w32', pretrained=True, features_only=True, out_indices=(0,1,2,3))
        chs = self.backbone.feature_info.channels()  # e.g., [32, 64, 128, 256]

        # Fuse HRNet multi-res to highest resolution branch via upsample+concat+1x1 (not "lite"; keeps detail)
        fused_out = 512
        self.fuse_convs = nn.ModuleList([nn.Conv2d(c, fused_out // len(chs), kernel_size=1, bias=False) for c in chs])
        self.fuse_bnrelus = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(fused_out // len(chs)), nn.ReLU(inplace=True))
                                           for _ in chs])
        self.post_fuse = nn.Sequential(
            nn.Conv2d(fused_out, fused_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_out),
            nn.ReLU(inplace=True),
        )

        # OCR head on fused feature (stride 4 resolution)
        self.ocr = SpatialOCRModule(
            in_channels=fused_out,
            key_channels=self.ocr_key_ch, value_channels=self.ocr_val_ch,
            out_channels=self.ocr_out_ch, num_regions=self.context_regions
        )

        # Heads from OCR features (upsample to 1x at the end)
        head_in = self.ocr_out_ch
        self.head_lane = nn.Sequential(
            nn.Conv2d(head_in, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        self.head_dir = nn.Sequential(
            nn.Conv2d(head_in, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1)   # will be normalized to unit vectors
        )

    def _fuse_hrnet_feats(self, feats):
        """
        feats: list of 4 tensors from HRNet (low->high stride). Upsample to stride-4 map and fuse.
        return: [B, 512, H/4, W/4]
        """
        base_h, base_w = feats[0].shape[-2:]
        outs = []
        for x, conv, bnrelu in zip(feats, self.fuse_convs, self.fuse_bnrelus):
            if x.shape[-2:] != (base_h, base_w):
                x = F.interpolate(x, size=(base_h, base_w), mode='bilinear', align_corners=False)
            x = bnrelu(conv(x))
            outs.append(x)
        fused = torch.cat(outs, dim=1)
        fused = self.post_fuse(fused)
        return fused

    def forward(self, x):
        # HRNet multi-scale features
        feats = self.backbone(x)                 # strides [4,8,16,32]
        fused = self._fuse_hrnet_feats(feats)    # stride-4

        # OCR context enhancement
        ocr_feats = self.ocr(fused)              # stride-4, rich context

        # Upsample to full resolution once for all heads
        ocr_full = F.interpolate(ocr_feats, scale_factor=4, mode='bilinear', align_corners=False)

        # Heads
        lane_logits = self.head_lane(ocr_full)   # [B,2,H,W]
        dir_map     = self.head_dir(ocr_full)    # [B,2,H,W]

        return lane_logits, dir_map
