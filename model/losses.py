import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ContrastiveLossInfoNCE(nn.Module):
    def __init__(self):
        super(ContrastiveLossInfoNCE, self).__init__()
        self.temperature = nn.Parameter(torch.ones([]).cuda() * np.log(1 / 0.07))
        self.eps = 1e-8

    def forward(self, image_features, text_features, labels=None):
        """
            image_embeddings: [B, 512]
            text_embeddings: [B, 512]
            labels: [B, au_nums]
        """
        # import pdb; pdb.set_trace()
        B, _ = image_features.shape
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_scale = self.temperature.exp()
        logits_per_image = logits_scale * image_features @ text_features.t() # [B, au_nums]
        mask = labels.bool()
        
        logits_per_image_log_softmax = F.log_softmax(logits_per_image, dim=-1) # Shape: [B, au_nums]
        
        positive_log_probs = torch.where(mask, logits_per_image_log_softmax, torch.tensor(0).cuda()) # Shape: [B, au_nums]
        per_image_positive_nll_sum = -positive_log_probs.sum(dim=1) # Shape: [B]
        num_positives_per_image  = labels.sum(dim=1) # Shape: [B]
        
        # --- **防止除零错误** ---
        # 识别出哪些图像至少有一个正样本
        valid_images_mask = num_positives_per_image > 0 # 形状: [batch_size]
        
        if not valid_images_mask.any():
            return torch.tensor(0.0, device=image_features.device, dtype=image_features.dtype, requires_grad=True)
        
        # 仅对包含正样本的图像计算平均损失
        # `valid_images_mask` 用于过滤掉没有正样本的图像，避免除以零
        valid_per_image_nll_sum = per_image_positive_nll_sum[valid_images_mask]
        valid_num_positives = num_positives_per_image[valid_images_mask]
        
        # 计算每个有效图像的平均正样本 NLL (损失)
        # 因为已经过滤，valid_num_positives 保证 > 0，但为保险起见或处理极小值，可加 eps
        average_nll_per_valid_image = valid_per_image_nll_sum / (valid_num_positives + self.eps)
        loss = average_nll_per_valid_image.mean()
        
        # print('temperature:', self.temperature)
        
        return loss
        
        
        
class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y):
        """
            x: pred [B, 8]
            y: true label [B, 8]
        """
        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        return -loss.mean()