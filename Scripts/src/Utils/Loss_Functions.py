import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedFocalLabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(CombinedFocalLabelSmoothingLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        if logits.size() != targets.size():
            raise ValueError("Logits and targets must be of the same shape.")

        smoothed_targets = targets * (1 - self.smoothing) + self.smoothing * 0.5

        bce_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction='none')

        probas = torch.sigmoid(logits)

        p_t = probas * smoothed_targets + (1 - probas) * (1 - smoothed_targets)
        alpha_t = self.alpha * smoothed_targets + (1 - self.alpha) * (1 - smoothed_targets)

        focal_weight = alpha_t * ((1 - p_t).pow(self.gamma))

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class CombinedFocalLabelSmoothingLossMultiCat(nn.Module):
    def __init__(self, alpha, gamma, smoothing=0.1, reduction='mean'):
        super(CombinedFocalLabelSmoothingLossMultiCat, self).__init__()
        self.alpha = alpha  # Should be a tensor of shape [num_classes]
        self.gamma = gamma  # Should be a tensor of shape [num_classes]
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        smoothed_targets = targets * (1 - self.smoothing) + self.smoothing * 0.5
        BCE_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction='none')

        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        epsilon = 1e-7
        p_t = torch.clamp(p_t, epsilon, 1. - epsilon)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        gamma_t = self.gamma * targets + self.gamma * (1 - targets)

        focal_loss = alpha_t * ((1 - p_t) ** gamma_t) * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss