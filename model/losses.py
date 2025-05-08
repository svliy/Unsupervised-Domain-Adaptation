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
    
def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).cuda() @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).cuda() @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss

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
    
    
def get_off_diagonal_elements(M):
    res = M.clone()
    res.diagonal(dim1=-1, dim2=-2).zero_()
    return res

class ContrastiveLossForSourceTm(nn.Module):
    """
    图像中心 (Image-Centric) 的对比损失函数。

    对于批次中的每一张图像 `i`:
    - 遍历该图像的所有 AU 特征 `image_features[i, k]`.
    - 计算 `image_features[i, k]` 与 *所有* 文本 AU 特征 `text_features[j]` 的相似度。
    - 如果 AU `k` 在图像 `i` 中是激活的 (`labels[i, k] == 1`):
        - 我们希望 `image_features[i, k]` 与对应的 `text_features[k]` (正样本) 的相似度最高。
        - 我们希望 `image_features[i, k]` 与其他的 `text_features[j]` (j != k, 负样本) 的相似度较低。
    - 这可以通过对每个激活的 `(i, k)` 对应用交叉熵损失来实现，其中目标类别是 `k`。

    优化：
    - 使用 einsum 计算所有 image_feature[i, k] vs text_feature[j] 的相似度。
    - 使用向量化的交叉熵损失计算，并通过标签进行掩码。
    """
    def __init__(self, initial_temperature=0.1, eps=1e-8):
        """
        Args:
            initial_temperature (float): 初始温度缩放因子。
            eps (float): 用于数值稳定性的小 epsilon 值。
        """
        super(ContrastiveLossForSourceTm, self).__init__()
        # 可学习的对数温度参数
        self.log_temperature = nn.Parameter(torch.ones([]) * np.log(1 / initial_temperature))
        self.eps = eps

    def forward(self, image_features, text_features, labels):
        """
        计算图像中心的对比损失。

        Args:
            image_features (torch.Tensor): 图像 AU 特征。形状: [B, num_classes, dim]
            text_features (torch.Tensor): 文本 AU 特征。形状: [num_classes, dim]
            labels (torch.Tensor): AU 激活标签 (0 或 1)。形状: [B, num_classes]

        Returns:
            torch.Tensor: 计算得到的对比损失（标量）。
        """
        # print(f'可学习温度系数：{self.log_temperature}')
        # --- 1. 输入验证与维度获取 ---
        assert image_features.ndim == 3 and text_features.ndim == 2 and labels.ndim == 2
        assert image_features.shape[0] == labels.shape[0]
        assert image_features.shape[1] == text_features.shape[0] == labels.shape[1]
        assert image_features.shape[2] == text_features.shape[1]

        B, num_classes, dim = image_features.shape
        device = image_features.device

        # --- 2. 特征归一化 ---
        # L2 归一化以计算余弦相似度
        image_features = F.normalize(image_features, p=2, dim=-1, eps=self.eps) # [B, C, D]
        text_features = F.normalize(text_features, p=2, dim=-1, eps=self.eps)   # [C, D]

        # --- 3. 计算图像AU特征 vs 所有文本AU特征的相似度 ---
        # 我们需要计算 sim(image[i, k], text[j]) for all i, k, j
        # 使用 einsum: 'b k d, j d -> b k j' (b=batch, k=img_au_idx, d=dim, j=txt_au_idx)
        # 或者使用矩阵乘法: image_features @ text_features.T
        # result[i, k, j] = similarity between image_features[i, k] and text_features[j]
        similarities = torch.einsum('bkd,jd->bkj', image_features, text_features) # 形状: [B, num_classes, num_classes]

        # --- 4. 应用温度缩放 ---
        temperature = torch.exp(self.log_temperature)
        logits = similarities / temperature # 形状: [B, num_classes, num_classes]
        # logits[i, k, :] 代表 image_features[i, k] 与所有 text_features[:] 的相似度（已缩放）

        # --- 5. 构建交叉熵损失的目标和输入 ---
        positive_logits = torch.diagonal(logits, offset=0, dim1=1, dim2=2) # [B, num_classes] - 最简洁！
        
        # 1. 向量化计算 positive values
        # 原代码中 masked_loss = positive_logits_i * labels_tmp
        # 在 batch 维度上向量化就是 element-wise multiplication
        masked_loss_batch = positive_logits * labels # 形状: (B, N)

        # 原代码中 pos_values = torch.sum(masked_loss) (per instance)
        # 在 batch 维度上向量化就是对最后一个维度求和
        pos_values_batch = torch.sum(masked_loss_batch, dim=1) # 形状: (B,)

        # 2. 向量化计算 num_positives
        # 原代码中 num_positives = torch.sum(labels_tmp) (per instance)
        num_positives_batch = torch.sum(labels, dim=1) # 形状: (B,)

        # 3. 向量化处理 num_positives < self.eps 的情况
        # 使用 torch.where 根据条件选择值
        # 如果 num_positives < self.eps，则 pos_values 设置为 0，否则保持原值
        # 需要确保 0.0 的 dtype 与 pos_values_batch 匹配
        pos_values_batch_adjusted = torch.where(
            num_positives_batch < self.eps,
            torch.tensor(0.0, device=device, dtype=pos_values_batch.dtype), # 使用 tensor 而不是 float，并指定 device 和 dtype
            pos_values_batch
        )

        # 4. 向量化计算 negative values (等效于 get_off_diagonal_elements 的 batch 版本)
        # 假设 logits 形状为 (B, N, N)
        # 负样本值通常是除了正样本（对角线）以外的其他 logit 值
        # 计算每个样本的对角线元素之和
        diag_elements_sum_batch = torch.sum(torch.diagonal(logits, dim1=1, dim2=2), dim=1) # 形状: (B,)

        # 计算每个样本的所有元素之和
        total_elements_sum_batch = torch.sum(logits, dim=(1, 2)) # 形状: (B,)

        # 非对角线元素之和 = 所有元素之和 - 对角线元素之和
        # 这等效于原始代码中的 neg_values = get_off_diagonal_elements(all_logits)，假设它计算的是非对角线总和
        neg_values_batch = total_elements_sum_batch - diag_elements_sum_batch # 形状: (B,)

        # 5. 向量化计算每个样本的 loss_i
        # loss_i = pos_values / (pos_values + neg_values)
        # 为了数值稳定性，分母加上一个小的 epsilon
        denominator = pos_values_batch_adjusted + neg_values_batch + self.eps # 形状: (B,)
        # 确保分母不是接近于零导致 NaN 或 Inf。虽然加了 self.eps，但如果 pos_values 和 neg_values 都很大负，也可能出问题。
        # 但根据通常的相似度或 logit 输出，它们不太会是很大的负数，除非 logits 本身有问题。
        # 另外，如果分子 pos_values_batch_adjusted 是 0，分母非零，结果是 0，这是合理的。
        # 如果分子非零，分母为零（理论上加了eps不会是0），则会导致 Inf，这是不合理的。
        # 如果分子和分母都为零（例如，没有正样本且非对角线总和为0），结果是 NaN。
        # 原始代码在 pos_values 为 0 时分子是 0，如果 neg_values 也为 0，分母为 0，也会是 NaN。
        # 加入 eps 提高了稳定性，但极端情况仍需注意输入的 logit 值范围。

        # 计算每个样本的 loss
        loss_batch = pos_values_batch_adjusted / denominator # 形状: (B,)
        # import pdb; pdb.set_trace()
        # 6. 计算整个 batch 的总 loss
        # 原代码是简单相加，向量化后就是对 batch 维度求和
        loss_all = torch.sum(loss_batch) / B

        # 返回总 loss
        return loss_all # 这个返回值和原始代码一致
    
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