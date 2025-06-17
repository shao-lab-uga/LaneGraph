import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric

class LaneAndDirectionExtractionLoss():
    
    def __init__(self, device, config):
        
        self.device = device
        # Lane
        self.lane_cross_entropy_loss = MeanMetric().to(self.device)
        self.lane_dice_loss = MeanMetric().to(self.device)
        self.lane_cross_entropy_loss_weight = config.lane_cross_entropy_loss_weight
        self.lane_dice_loss_weight = config.lane_dice_loss_weight
        # Direction
        self.direction_l2_loss = MeanMetric().to(self.device)
        self.direction_l2_loss_weight = config.direction_l2_loss_weight

    def update(self, loss_dict):
        """
        Update the loss metrics with the provided loss dictionary.

        Args:
            loss_dict: Dictionary containing loss values.
        """
        self.lane_cross_entropy_loss.update(loss_dict['lane_cross_entropy_loss'])
        self.lane_dice_loss.update(loss_dict['lane_dice_loss'])
        self.direction_l2_loss.update(loss_dict['direction_l2_loss'])
    
    def get_result(self):
        """
        Get the computed loss metrics.

        Returns:
            Dictionary containing the computed loss metrics.
        """
        result_dict = {}
        result_dict['lane_cross_entropy_loss'] = self.lane_cross_entropy_loss.compute()
        result_dict['lane_dice_loss'] = self.lane_dice_loss.compute()
        result_dict['direction_l2_loss'] = self.direction_l2_loss.compute()
        
        # Reset the metrics after computation
        self.reset()
        
        return result_dict
    
    def compute(self, lane_predicted, direction_predicted, region_mask, lane_groundtruth, direction_groundtruth):
        """
        loss function for lane and direction extraction.
        Args:
            lane_predicted: [B, 2, H, W] (raw logits for lane)
            direction_predicted: [B, 2, H, W] (raw logits for direction)
            region_mask: [B, 1, H, W] (mask to include/exclude pixels)
            lane_groundtruth: [B, 1, H, W] (ground truth for lane)
            direction_groundtruth: [B, 2, H, W] (ground truth for direction)
        Returns:
            loss: scalar tensor representing the total loss
        """
        lane_cross_entropy_loss = self.binary_softmax_ce_loss(lane_predicted, lane_groundtruth, region_mask)
        lane_dice_loss = self.binary_dice_loss(lane_predicted, lane_groundtruth, region_mask)
        direction_l2_loss = torch.mean(region_mask * torch.square(direction_groundtruth - direction_predicted))
        # Combine losses into a dictionary
        loss_dict = {
            'lane_cross_entropy_loss': lane_cross_entropy_loss * self.lane_cross_entropy_loss_weight,
            'lane_dice_loss': lane_dice_loss * self.lane_dice_loss_weight,
            'direction_l2_loss': direction_l2_loss * self.direction_l2_loss_weight
        }
        
        return loss_dict

    def reset(self):
        """
        Reset the loss metrics.
        """
        self.lane_cross_entropy_loss.reset()
        self.lane_dice_loss.reset()
        self.direction_l2_loss.reset()

    def binary_softmax_ce_loss(self, logits, targets, mask):
        """
        Binary softmax cross-entropy loss using two-channel logits.
        logits: [B, 2, H, W] (raw outputs)
        targets: [B, 1, H, W] (0 or 1)
        mask: [B, 1, H, W] (1: include, 0: ignore)
        """
        logsumexp = F.logsumexp(logits, dim=1, keepdim=True)  # [B, 1, H, W]
        class0 = logits[:, 0:1, :, :]
        class1 = logits[:, 1:2, :, :]
        loss = -(targets * class0 + (1 - targets) * class1 - logsumexp)
        return torch.mean(loss * mask)


    def binary_dice_loss(self, logits, targets, mask):
        """
        Dice loss computed using sigmoid on the difference of logits.
        logits: [B, 2, H, W]
        targets: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        probs = torch.sigmoid(logits[:, 0:1, :, :] - logits[:, 1:2, :, :])
        intersection = torch.sum(probs * targets * mask)
        union = torch.sum((probs + targets) * mask) + 1.0
        return 1.0 - (2.0 * intersection / union)