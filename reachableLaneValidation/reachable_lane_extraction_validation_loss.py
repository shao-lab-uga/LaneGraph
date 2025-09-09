import torch
from torchmetrics import MeanMetric

class ReachableLaneExtractionValidationLoss():
    
    def __init__(self, device, config):
        
        self.device = device
        # Lane
        self.lane_cross_entropy_loss = MeanMetric().to(self.device)
        self.lane_dice_loss = MeanMetric().to(self.device)
        # Label
        self.reachable_label_cross_entropy_loss = MeanMetric().to(self.device)

        self.lane_cross_entropy_loss_weight = config.lane_cross_entropy_loss_weight
        self.lane_dice_loss_weight = config.lane_dice_loss_weight
        self.reachable_label_cross_entropy_loss_weight = config.reachable_label_cross_entropy_loss_weight

    def update(self, loss_dict):
        """
        Update the loss metrics with the provided loss dictionary.

        Args:
            loss_dict: Dictionary containing loss values.
        """
        self.lane_cross_entropy_loss.update(loss_dict['lane_cross_entropy_loss'])
        self.lane_dice_loss.update(loss_dict['lane_dice_loss'])
        self.reachable_label_cross_entropy_loss.update(loss_dict['reachable_label_cross_entropy_loss'])
    
    def get_result(self):
        """
        Get the computed loss metrics.

        Returns:
            Dictionary containing the computed loss metrics.
        """
        result_dict = {}
        result_dict['lane_cross_entropy_loss'] = self.lane_cross_entropy_loss.compute()
        result_dict['lane_dice_loss'] = self.lane_dice_loss.compute()
        result_dict['reachable_label_cross_entropy_loss'] = self.reachable_label_cross_entropy_loss.compute()
        
        # Reset the metrics after computation
        self.reset()
        
        return result_dict
    
    def compute(self, reachable_lane_predicted_a, reachable_lane_groundtruth_a, reachable_lane_predicted_b, reachable_lane_groundtruth_b, reachable_label_predicted, reachable_label_groundtruth):
        """
        loss function for lane and direction extraction.
        Args:

            reachable_lane_predicted_a: [B, 2, H, W] (raw logits for lane A)
            reachable_lane_predicted_b: [B, 2, H, W] (raw logits for lane B)
            reachable_lane_groundtruth_a: [B, 1, H, W] (ground truth for lane A)
            reachable_lane_groundtruth_b: [B, 1, H, W] (ground truth for lane B)
            reachable_label_predicted: [B, 2] (raw logits for reachable label)
            reachable_label_groundtruth: [B, 1] (ground truth for reachable label)
        Returns:
            loss: scalar tensor representing the total loss
        """
        
        
        # Compute the dice loss for lane predictions
        lane_dice_loss = self.dice_loss(reachable_lane_predicted_a, reachable_lane_groundtruth_a) + self.dice_loss(reachable_lane_predicted_b, reachable_lane_groundtruth_b)
        lane_cross_entropy_loss = self.cross_entropy_loss(reachable_lane_predicted_a, reachable_lane_groundtruth_a) + self.cross_entropy_loss(reachable_lane_predicted_b, reachable_lane_groundtruth_b)
        # Combine losses into a dictionary
        
        loss_dict = {
            'lane_cross_entropy_loss': lane_cross_entropy_loss * self.lane_cross_entropy_loss_weight,
            'lane_dice_loss': lane_dice_loss * self.lane_dice_loss_weight,
            'reachable_label_cross_entropy_loss': self.cross_entropy_loss(reachable_label_predicted, reachable_label_groundtruth) * self.reachable_label_cross_entropy_loss_weight
        }
        
        return loss_dict

    def reset(self):
        """
        Reset the loss metrics.
        """
        self.lane_cross_entropy_loss.reset()
        self.lane_dice_loss.reset()
        self.reachable_label_cross_entropy_loss.reset()


    def cross_entropy_loss(self, logits, targets):
        """
        Custom cross-entropy loss for binary classification with two logits channels.
        logits: prediction logits of shape [B, 2, H, W]
        targets: target labels of shape [B, 1, H, W], binary (0 or 1)
        """
        p0 = logits[:, 0:1]  # logits for class 0 which is the lane
        p1 = logits[:, 1:2]  # logits for class 1 which is the non-lane

        # Numerically stable log-sum-exp formulation of softmax cross-entropy
        logsumexp = torch.logsumexp(logits, dim=1, keepdim=True)
        loss = -(targets * p0 + (1 - targets) * p1 - logsumexp)

        return torch.mean(loss)


    def dice_loss(self, logits, targets):
        """
        Dice loss based on the sigmoid of logit difference.
        logits: prediction logits of shape [B, 2, H, W]
        targets: target labels of shape [B, 1, H, W]
        """
        # Convert logits to probability map using sigmoid on class difference
        prob = torch.sigmoid(logits[:, 0:1, :, :] - logits[:, 1:2, :, :])

        numerator = 2 * torch.sum(prob * targets)
        denominator = torch.sum((prob + targets)) + 1.0  # avoid zero division

        return 1 - numerator / denominator