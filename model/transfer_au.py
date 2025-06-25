import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb
from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .graph import GNN
from .description import describe_au
from .transformers_encoder.transformer import TransformerEncoder as CrossAttention
from .losses import WeightedAsymmetricLoss, ContrastiveLossInfoNCE, MMD_loss, ContrastiveLossForSource

_tokenizer = _Tokenizer()

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features=None,drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        """
        input: the shape of x is [B, C, H, W]
        output: the shape of x is [B, C, H, W]
        """
        # x: [256, 2048, 7, 7]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1) # [B, 49, 2048]
        x = self.drop(x) # [B, 49, 2048]
        x = self.fc(x).permute(0, 2, 1) # [B, 512, 49]
        # the input of BN_1D is (N, C, L)
        x = self.relu(self.bn(x)).permute(0, 2, 1) # [B, 49, 512]
        B, C, _, _ = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class VisionAdapter(nn.Module):
    """
    该模块包含两个残差连接和一个多路并行卷积块。
    """
    def __init__(self, in_dim, out_dim, pool_kernel, pool_stride):
        
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm2d(out_dim)

        self.conv3x3_1 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.conv3x3_2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.conv3x3_3 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        
        B, C_in, H, W = x.shape # [256, 2048, 7, 7]
        x = x.reshape(B, C_in, H*W).permute(0, 2, 1) # [B, H*W, 2048]
        x = self.projection(x).permute(0, 2, 1) # [B, 256, H*W]
        
        B, C_out, _ = x.shape
        x = x.reshape(B, C_out, H, W)
        identity_1 = x
        x = self.bn(x)
        x = x + identity_1

        # 多路并行卷积
        identity_2 = x
        conv3_out = self.conv3x3_1(x) # [B, 256, H, W]
        conv5_out = self.conv3x3_2(x) # [B, 256, H, W]
        conv7_out = self.conv3x3_3(x) # [B, 256, H, W]
        # 平均卷积结果
        avg_conv_out = (conv3_out + conv5_out + conv7_out) / 3.0
        x = avg_conv_out + identity_2

        x = self.relu(x)
        x = self.pool(x)
        return x

# class VisionAdapter(nn.Module):
#     """Applies a 1x1 Conv, BatchNorm, ReLU, and AvgPool."""
#     def __init__(self, in_dim, out_dim, pool_kernel, pool_stride):
#         super().__init__()
#         self.adapter = nn.Sequential(
#             nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(inplace=True)
#         )
#         # Ensure pool output size is consistent if needed, e.g., target 7x7
#         # Simple AvgPool is used here as in the original code.
#         self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

#     def forward(self, x):
#         return self.pool(self.adapter(x))

class BranchLayer(nn.Module):
    """
    一个带残差连接的特征瓶颈层 (Bottleneck Layer with Residual Connection)。
    它应用 C -> C/2 -> C 的变换，然后将结果与原始输入相加。
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        # 计算瓶颈部分的维度
        bottleneck_dim = feature_dim // 2
        # 使用 nn.Sequential 构建瓶颈结构，使代码更清晰
        self.bottleneck = nn.Sequential(
            # 1. 压缩层：feature_dim -> feature_dim // 2
            nn.Conv2d(feature_dim, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            
            # 2. 扩张层：feature_dim // 2 -> feature_dim
            nn.Conv2d(bottleneck_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
        )
        # 单独定义最终的激活函数
        self.final_relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        定义前向传播过程，包含残差连接。
        """
        # 1. 保存原始输入，作为残差连接的 "identity"
        identity = x
        # 2. 将输入通过瓶颈变换结构 F(x)
        residual = self.bottleneck(x)
        # 3. 将瓶颈结构的输出与原始输入相加
        output = identity + residual
        # 4. 在相加后应用最终的激活函数
        output = self.final_relu(output)
        
        return output

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # 此时prompt已经是词向量形式了
        # text前向传播合batch size 无关？对，只和 class数量有关
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    
    def __init__(self, cfg, classnames=None, clip_model=None):
        
        super().__init__()
        n_cls = len(classnames) # 类别数量
        n_ctx = cfg.n_ctx # 可学习上下文token的数量
        ctx_init = cfg.ctx_init # 
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0] # 上下文向量的维度：512
        clip_imsize = clip_model.visual.input_resolution # 224
        cfg_imsize = cfg.input_shape[1] # 输入数据的图像尺寸
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            ##### 执行这个分支
            
            # random initialization
            if cfg.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                # 执行该分支
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # [16, 512]
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx) # 'X X X X X X X X X X X X X X X X'

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames] # 句号重复

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # [10, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # [10, 77, 512]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        
        # 将所有提示共享的起始符 ([SOS] - Start of Sequence) token 的嵌入存储为一个 buffer
        # Buffer 是模型状态的一部分，会被保存，但不会被优化器更新（因为它是固定的）。
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS [10, 1, 512]
        # 将类别名称 token 和结束符 ([EOS] - End of Sequence) token 的嵌入存储为另一个 buffer。同样，这部分也是固定的。
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS # [10, 60, 512]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.ctp

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # [10, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class TransferNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.class_num
        self.vision_dim = 256
        
        clip_model, _ = clip.load(config.backbone, device="cpu")
        # CoOp
        # text = describe_au(config.au_list)
        # self.prompt_learner = PromptLearner(config, classnames=text, clip_model=clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # Encoder for Vision and Language
        self.image_encoder = clip_model.visual
        # self.text_encoder = TextEncoder(clip_model)
        # CLIP
        # self.clip_logit_scale = clip_model.logit_scale
        # self.dtype = clip_model.dtype
        # 文本
        # self.text_transform = nn.Linear(self.vision_dim*4, self.vision_dim)
        
        print("冻结CLIP模型视觉和文本编码器")
        # for param in list(self.image_encoder.parameters()) + list(self.text_encoder.parameters()):
        #     param.requires_grad = False
        for param in list(self.image_encoder.parameters()):
            param.requires_grad = False
        # self.clip_logit_scale.requires_grad = False
        
        # 适配器
        self.adapter_s1 = VisionAdapter(self.vision_dim , self.vision_dim, pool_kernel=8, pool_stride=8) # 56x56 -> 7x7
        self.adapter_s2 = VisionAdapter(self.vision_dim * 2, self.vision_dim, pool_kernel=4, pool_stride=4) # 28x28 -> 7x7
        self.adapter_s3 = VisionAdapter(self.vision_dim * 4, self.vision_dim, pool_kernel=2, pool_stride=2) # 14x14 -> 7x7
        self.adapter_s4 = VisionAdapter(self.vision_dim * 8, self.vision_dim, pool_kernel=1, pool_stride=1) # 7x7 -> 7x7

        # Cross Attention
        # (seq_len, batch, embed_dim)
        # self.cross_attention = CrossAttention(embed_dim=self.vision_dim, num_heads=8, layers=2)
        
        self.branch_layers = nn.ModuleList([
            BranchLayer(self.vision_dim) for _ in range(self.num_classes)
        ])
        
        # 图神经网络
        self.gnn = GNN(self.vision_dim, num_classes=self.num_classes, neighbor_num=config.neighbor_num, metric='cosine')

        # --- Independent AU Classifiers ---
        # Create a separate linear classifier for each AU
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.vision_dim, 1))
            for _ in range(self.num_classes)
        ])
        
        
    def freeze_bn(self):
        # 冻结图像编码器的BN层
        for module in self.image_encoder.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
    
    def forward(self, image, text=None, labels=None):

        B, C, H, W = image.shape # images: [128, 3, 224, 224]

        # 1. 提取视觉特征
        image_features = self.image_encoder(image) # RN50: [128, 1024]
        image_features_s1 = image_features['layer1'] # Shape: [B, 256, 56, 56]
        image_features_s2 = image_features['layer2'] # Shape: [B, 512, 28, 28]
        image_features_s3 = image_features['layer3'] # Shape: [B, 1024, 14, 14]
        image_features_s4 = image_features['layer4'] # Shape: [B, 2048, 7, 7]

        # 2. 适配器
        adapter_features_s1 = self.adapter_s1(image_features_s1) # Shape: [B, 256, 7, 7]
        adapter_features_s2 = self.adapter_s2(image_features_s2) # Shape: [B, 256, 7, 7]
        adapter_features_s3 = self.adapter_s3(image_features_s3) # Shape: [B, 256, 7, 7]
        adapter_features_s4 = self.adapter_s4(image_features_s4) # Shape: [B, 256, 7, 7]
        # 特征相加: [B, 256, 7, 7]
        image_features_fusion = adapter_features_s1 + adapter_features_s2 + adapter_features_s3 + adapter_features_s4

        # 1. text features
        # prompts = self.prompt_learner() # [10, 77, 512]
        # tokenized_prompts = self.tokenized_prompts # [10, 77]
        # text_features = self.text_encoder(prompts, tokenized_prompts) # [10, 1024]
        # text_features = self.text_transform(text_features) # [10, 256]
        
        
       
        
        
        ## 2.3 Combine Multi-Scale Features
        # image_features_fusion = torch.cat([image_features_s1, image_features_s2, image_features_s3, image_features_s4], dim=1) # Shape: [B, 1024, 7, 7]
        # image_features_fusion = self.multi_scale_fusion(image_features_fusion) # Shape: [B, 256, 7, 7]
        
        # # 3. cross attention
        # # combined_features
        # B, cross_dim, H, W = combined_features.shape
        # combined_features_cross = combined_features.reshape(B, cross_dim, H*W).permute(2, 0, 1) # [49, 256, 256]
        # text_features_cross = text_features.unsqueeze(1).expand(-1, B, -1) # [10, B, 256]
        # # (seq_len, batch, embed_dim) [49, 256, 1024]
        # vision_features_cross = self.cross_attention(combined_features_cross, text_features_cross, text_features_cross)
        # vision_features_cross = vision_features_cross.permute(1, 2, 0).reshape(B, cross_dim, H, W)
        # vision_features_cross = vision_features_cross + combined_features # [B, 256, 7, 7]
        
        # vision_features_cross = image_features_s4
        
        # image_features_fusion = image_features_s4

        # 4. 多分支AU网络

        # 使用列表推导式高效地得到所有分支的输出, 结果是一个list，包含N个 [B, C, H, W] 的张量
        branch_outputs = [layer(image_features_fusion) for layer in self.branch_layers]
        # 新张量的形状为 [B, N, C, H, W]，其中 N 是分支数量
        au_specific_features = torch.stack(branch_outputs, dim=1) 
        au_specific_features = au_specific_features.reshape(B, self.config.class_num, self.vision_dim, -1)
        au_pooled_features = au_specific_features.mean(dim=-1) # [B, num_classes, 256]
        
        # 5. 图卷积网络
        gnn_features = self.gnn(au_pooled_features) # Shape: [B, num_classes, 256]
        
        # 6. AU分类器
        logits = []
        for i in range(self.num_classes):
            au_feature = gnn_features[:, i, :] # Shape: [B, vision_dim]
            au_feature = F.normalize(au_feature, p=2, dim=-1)
            logit = self.classifiers[i](au_feature) # Shape: [B, 1]
            logits.append(logit)
        logits = torch.cat(logits, dim=1) # Shape: [B, num_classes]
        
        return {
            'logits': logits,
            # 'au_vision_features': gnn_features, # [B, num_classes, 256]
            # 'au_text_features': text_features,  # [num_classes, 256]
        }
    
if __name__ == "__main__":
    pass