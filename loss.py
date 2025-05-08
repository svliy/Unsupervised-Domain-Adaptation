import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pdb

class ContrastiveLossInfoNCE(nn.Module):
    def __init__(self):
        super(ContrastiveLossInfoNCE, self).__init__()
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
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
    
class TextCenterInfoNCE(nn.Module):
    def __init__(self):
        super(TextCenterInfoNCE, self).__init__()
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))
        self.eps = 1e-8

    def forward(self, image_features, text_features, labels=None):
        """
            image_embeddings: [B, 512]
            text_embeddings: [au_nums, 512]
            labels: [B, au_nums]
        """
        # Input shapes
        B, _ = image_features.shape
        num_aus, text_dim = text_features.shape
        
        # --- 1. Normalization ---
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # --- 2. Calculate Logits ---
        # Logits represent the similarity between each AU and each image.
        # We want AU j's similarity to all images i = 1...B.
        # Shape: [num_aus, B]
        logits_scale = self.temperature.exp()
        logits_per_au = logits_scale * text_features @ image_features.t()
        
        # --- 3. Prepare Mask ---
        # The mask indicates positive pairs (AU j, Image i) where AU j is active in image i.
        # We need the transpose of the input labels.
        # Shape: [num_aus, B]
        positive_mask = labels.t().bool() # Transpose to align with logits_per_au
        
        
        # --- 4. Calculate Log Softmax ---
        log_probs = F.log_softmax(logits_per_au, dim=1) # Shape: [num_aus, B]
        
        # --- 5. Select Log Probs for Positive Pairs ---
        positive_log_probs = torch.where(positive_mask, log_probs, torch.zeros_like(log_probs)) # Shape: [num_aus, B]
        
        # --- 6. Calculate NLL per AU ---
        per_au_positive_nll_sum = -positive_log_probs.sum(dim=1) # Shape: [num_aus]
        
        # --- 7. Count Positive Images per AU ---
        num_positives_per_au = labels.sum(dim=0) # Shape: [num_aus]
        
        # --- 8. Handle AUs with No Positive Samples ---
        # Identify AUs that have at least one positive image in the batch.
        valid_aus_mask = num_positives_per_au > 0 # Shape: [num_aus]
        # 识别出哪些图像至少有一个正样本
        
        if not valid_aus_mask.any():
            return torch.tensor(0.0, device=image_features.device, dtype=image_features.dtype, requires_grad=True)
        
        # --- 9. Calculate Average NLL for Valid AUs ---
        # Filter out AUs that had no positive samples in the batch.
        valid_per_au_nll_sum = per_au_positive_nll_sum[valid_aus_mask]
        valid_num_positives = num_positives_per_au[valid_aus_mask]
        average_nll_per_valid_au = valid_per_au_nll_sum / (valid_num_positives + self.eps)
        loss = average_nll_per_valid_au.mean()
        
        # print('temperature:', self.temperature)
        return loss

class VisionTextContrastiveLoss(nn.Module):
    """
    对比学习损失，不使用标签进行过滤。
    正样本对: (visual_features[i, j, :], textual_features[j, :])
    对所有 i, j 计算损失。
    输入:
        visual_features: 视觉特征张量，形状为 [B, au_nums, feature_dim]，
                         其中 B 是批次大小, au_nums 是 Action Unit 的数量, 
                         feature_dim 是特征维度 (例如 512)。
                         visual_features[i, j, :] 代表图像 i 中 AU j 的视觉特征。
        textual_features: 文本特征张量，形状为 [au_nums, feature_dim]。
                          textual_features[j, :] 代表 AU j 的文本特征。
        labels: 标签张量，形状为 [B, au_nums]。 labels[i, j] = 1 表示 AU j 
                在图像 i 中是活跃的，否则为 0。用于过滤计算损失的样本。

    对比逻辑:
        - 正样本对: (visual_features[i, j, :], textual_features[j, :])
        - 负样本对: (visual_features[i, j, :], textual_features[k, :]) 其中 k != j
        - 仅对标签中标记为活跃的 AU (即 labels[i, j] == 1) 计算损失。
    """
    def __init__(self, initial_temperature=0.1):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(np.log(1 / initial_temperature)))
        self.eps = 1e-8

    def forward(self, visual_features, textual_features, labels=None): # labels 参数保留但未使用
        B, num_aus, visual_dim = visual_features.shape
        _, text_dim = textual_features.shape
        assert visual_dim == text_dim
        feature_dim = visual_dim

        visual_features_norm = F.normalize(visual_features, p=2, dim=-1)
        textual_features_norm = F.normalize(textual_features, p=2, dim=-1)

        visual_features_flat = visual_features_norm.view(B * num_aus, feature_dim)
        logits = visual_features_flat @ textual_features_norm.t() # [B * au_nums, au_nums]
        logits_scaled = logits * self.temperature.exp()
        # 没有考虑不同样本之间的联系

        target_labels = torch.arange(num_aus, device=visual_features.device).repeat(B)

        # 直接计算所有样本的交叉熵损失
        loss = F.cross_entropy(logits_scaled, target_labels)
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
        # if self.disable_torch_grad:
            # torch.set_grad_enabled(False)
        # neg_weight = 1 - xs_neg
        # if self.disable_torch_grad:
            # torch.set_grad_enabled(True)
        loss = los_pos + los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        return -loss.mean()
    
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
        
class WeightedBCELoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedBCELoss, self).__init__()
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

        loss = los_pos + los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        return -loss.mean()
    
class GeneralizedCrossEntropy(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)

def BCEWithLogitsLoss_PNWeight(input, target, p_n_weight, weight=None, size_average=True, reduce=True):
    r"""Function that measures Binary Cross Entropy between target and output with p_n_weight
    logits.

    Modified from pytorch BCEWithLogitsLoss.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                :attr:`size_average` is set to ``False``, the losses are instead summed
                for each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on :attr:`size_average`. When :attr:`reduce`
                is ``False``, returns a loss per input/target element instead and ignores
                :attr:`size_average`. Default: ``True``

    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    # make all the negative value to positive and positive values to 0
    max_val = (-input).clamp(min=0)
    loss = (p_n_weight - 1) * target * (1 + (- input).exp()).log() + \
        input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    loss = torch.relu(loss)
    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

def get_off_diagonal_elements(M):
    res = M.clone()
    res.diagonal(dim1=-1, dim2=-2).zero_()
    return res

class ContrastiveLossForSource(nn.Module):
    """
    图像中心 (Image-Centric) 的对比损失函数 - 基于交叉熵。

    对于批次中的每一张图像 `i`:
    - 遍历该图像的所有 AU 特征 `image_features[i, k]`.
    - 计算 `image_features[i, k]` 与 *所有* 文本 AU 特征 `text_features[j]` 的相似度。
    - 如果 AU `k` 在图像 `i` 中是激活的 (`labels[i, k] == 1`):
        - 我们希望 `image_features[i, k]` 与对应的 `text_features[k]` (正样本) 的相似度最高。
        - 这可以通过对每个激活的 `(i, k)` 对应用交叉熵损失来实现，
          其中输入 logits 是 `sim(image[i, k], text[:])`，目标类别是 `k`。

    优化：
    - 使用 einsum 计算所有 image_feature[i, k] vs text_feature[j] 的相似度。
    - 使用向量化的交叉熵损失计算，并通过标签进行掩码和平均。
    """
    def __init__(self, initial_temperature=0.07, eps=1e-8): # CLIP 使用 0.07
        super().__init__()
        # 可学习的对数温度参数
        # 使用 log(1/T) 初始化，这样 exp(log_temp) = 1/T
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0 / initial_temperature)))
        self.eps = eps

    def forward(self, image_features, text_features, labels):
        """
        计算图像中心的对比损失 (交叉熵版本)。

        Args:
            image_features (torch.Tensor): 图像 AU 特征。形状: [B, num_classes, dim]
            text_features (torch.Tensor): 文本 AU 特征。形状: [num_classes, dim]
            labels (torch.Tensor): AU 激活标签 (0 或 1)。形状: [B, num_classes]

        Returns:
            torch.Tensor: 计算得到的对比损失（标量）。
        """
        assert image_features.ndim == 3 and text_features.ndim == 2 and labels.ndim == 2
        assert image_features.shape[0] == labels.shape[0]
        assert image_features.shape[1] == text_features.shape[0] == labels.shape[1]
        assert image_features.shape[2] == text_features.shape[1]

        B, num_classes, dim = image_features.shape
        device = image_features.device

        # --- 2. 特征归一化 ---
        image_features = F.normalize(image_features, p=2, dim=-1, eps=self.eps) # [B, C, D]
        text_features = F.normalize(text_features, p=2, dim=-1, eps=self.eps)   # [C, D]

        # --- 3. 计算图像AU特征 vs 所有文本AU特征的相似度 ---
        # similarities[i, k, j] = similarity between image_features[i, k] and text_features[j]
        similarities = torch.einsum('bkd,jd->bkj', image_features, text_features) # [B, C, C]

        # --- 4. 应用温度缩放 ---
        # 注意：CLIP论文中的 temperature T 是除数，等价于乘以 logit_scale = 1/T
        # 我们学习 log_temperature = log(1/T) = log(logit_scale)
        # 所以 logits = similarities * exp(log_temperature)
        # 或者 logits = similarities / exp(-log_temperature) -> T = exp(-log_temperature)
        # 或者 logits = similarities / T, 其中 T = exp(log_T) -> 这种需要改初始化 log(T)
        # 我们采用 CLIP 的方式：logits = similarities * logit_scale
        logit_scale = torch.exp(self.log_temperature)
        logits = similarities * logit_scale # [B, C, C]
        # logits[i, k, :] 代表 image_features[i, k] 与所有 text_features[:] 的相似度（已缩放）

        # --- 5. 构建交叉熵损失的目标和输入 ---
        # 目标：对于每个 image_features[i, k]，正确的 text_features 索引是 k
        targets = torch.arange(num_classes, device=device).long() # Shape: [C]
        # 为每个 batch 样本和每个图像 AU 特征 k 扩展目标
        # 我们需要对 B*C 个 image features 计算损失
        targets = targets.unsqueeze(0).expand(B, -1) # Shape: [B, C]
        targets_flat = targets.reshape(B * num_classes) # Shape: [B*C]

        # 输入 logits 需要调整形状为 [N, num_classes]
        # logits[i, k, :] 是 image_features[i, k] 与所有 text_features 的相似度
        logits_flat = logits.view(B * num_classes, num_classes) # Shape: [B*C, C]

        # 计算交叉熵损失 (不进行 reduction，以便后续掩码)
        loss_ce = F.cross_entropy(logits_flat, targets_flat, reduction='none') # Shape: [B*C]

        # 将损失 reshape 回 [B, C] 以便使用 labels 进行掩码
        loss_ce = loss_ce.view(B, num_classes) # Shape: [B, C]

        # --- 6. 掩码和平均 ---
        # 只计算激活了的 AU 的损失 (labels == 1)
        labels_float = labels.float() # 确保 labels 是浮点数以便乘法
        masked_loss = loss_ce * labels_float

        # 计算有效的损失项数量 (激活的 AU 总数)
        num_active_aus = labels_float.sum()

        # 计算最终损失：对所有激活的 AU 的损失求和，然后除以激活 AU 的数量
        if num_active_aus < self.eps:
            # 如果批次中没有激活的 AU，返回 0 或一个小的惩罚项
            # 返回 0 通常是安全的，因为它不会产生梯度
            # 确保它 requires_grad=True 如果 log_temperature 需要被优化
            final_loss = torch.tensor(0.0, device=device, requires_grad=self.log_temperature.requires_grad)
        else:
            final_loss = masked_loss.sum() / num_active_aus

        # (可选) 添加对 log_temperature 范围的约束，防止其过大或过小
        # self.log_temperature.data.clamp_(min=np.log(1/100), max=np.log(1/0.01)) # 例如限制 T 在 0.01 到 100 之间

        return final_loss

if __name__ == '__main__':
    
    gce_loss = GeneralizedCrossEntropy(q=0.7)
    input = torch.randn(128, 10)
    target = torch.randn(128, 10)
    import pdb; pdb.set_trace()
    output = gce_loss(input, target)
    
    print(output.shape)
    print(output)