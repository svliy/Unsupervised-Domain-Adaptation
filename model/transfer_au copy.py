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
    def __init__(self, in_features,out_features=None,drop=0.0):
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
        B,C,H,W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1) # [B, 49, 2048]
        x = self.drop(x) # [B, 49, 512]
        x = self.fc(x).permute(0, 2, 1) # [B, 512, 49]
        x = self.relu(self.bn(x))
        _,C,_ = x.shape
        x = x.reshape(B,C,H,W)
        return x

class VisionAdapter(nn.Module):
    """Applies a 1x1 Conv, BatchNorm, ReLU, and AvgPool."""
    def __init__(self, in_dim, out_dim, pool_kernel, pool_stride):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        # Ensure pool output size is consistent if needed, e.g., target 7x7
        # Simple AvgPool is used here as in the original code.
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        return self.pool(self.adapter(x))

class VisionAdapterV2(nn.Module):
    """Applies a 1x1 Conv, BatchNorm, ReLU, and AvgPool."""
    def __init__(self, in_dim, out_dim, pool_kernel, pool_stride):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        # self.conv_down =  nn.Conv2d(in_dim, out_dim, kernel_size=1)
        # self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=3 // 2)
        # self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=5, padding=5 // 2)
        # self.conv3 = nn.Conv2d(out_dim, out_dim, kernel_size=7, padding=7 // 2)
        # self.conv_identity = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        
        self.projection.weight.data.normal_(0, math.sqrt(2. / out_dim))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        
    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1) # [B, 49, 2048]
        x = self.projection(x) # [B, 49, 512]
        x = x.permute(0, 2, 1) # [B, 512, 49]
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(B, -1, H, W)
        # conv1_x = self.conv1(x)
        # conv2_x = self.conv2(x)
        # conv3_x = self.conv3(x)
        # x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        # x = (conv1_x + conv2_x + conv3_x) / 3.0
        # x = self.conv_identity(x)
        # pdb.set_trace()
        return x

class MonaOp(nn.Module):
    
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x
    
class MonaRes(nn.Module):
    # [B, L, C]
    # [256, 197, 768]
    def __init__(self, in_dim, inter_dim=128, out_dim=None, pool=None, hw_shapes=None):
        super().__init__()

        self.project1 = nn.Linear(in_dim, inter_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(inter_dim, out_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(inter_dim)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        
        self.pooling = nn.AvgPool2d(kernel_size=pool[0], stride=pool[1])
        self.hw_shapes = hw_shapes

    def forward(self, x, hw_shapes=None):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1) # [B, H*W, C]
        
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = self.hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        
        # out = identity + project2
        out = project2
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        out = self.pooling(out)

        return out
    
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

class MultiScaleCombiner(nn.Module):
    """
    一个用于组合多尺度特征的模块。

    Args:
        vision_dim (int): 输入和输出特征的维度。
    """
    def __init__(self, vision_dim: int, bottle_dim: int, dropout=0.3):
        super().__init__()
        self.vision_dim = vision_dim
        self.conv1 = nn.Conv2d(vision_dim, bottle_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1  = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(bottle_dim, bottle_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottle_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2  = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv2d(bottle_dim, bottle_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottle_dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3  = nn.Dropout(dropout)
        
        # self.conv3 = nn.Conv2d(self.vision_dim, self.vision_dim, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.vision_dim)
        # self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，通道数应为 4 * vision_dim。

        Returns:
            torch.Tensor: 输出张量，通道数应为 vision_dim。
        """
        out = self.conv1(x) # [128, 128, 7, 7]
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        indentation = out

        out = self.conv3(out)
        out = self.bn3(out)
        out = indentation + out
        out = self.relu3(out)
        out = self.drop3(out)
        return out

class TransferNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.class_num
        self.vision_dim = 512
        
        clip_model, _ = clip.load(config.backbone, device="cpu")
        # CoOp
        text = describe_au(config.au_list)
        self.prompt_learner = PromptLearner(config, classnames=text, clip_model=clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # Encoder for Vision and Language
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        # CLIP
        self.clip_logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # 文本
        self.text_transform = nn.Linear(self.vision_dim*4, self.vision_dim)
        
        print("冻结CLIP模型视觉和文本编码器")
        for param in list(self.image_encoder.parameters()) + list(self.text_encoder.parameters()):
            param.requires_grad = False
        self.clip_logit_scale.requires_grad = False
        
        # Vision Adapters for multi-scale features
        # Assuming ResNet50 stages output: s1: 256, s2: 512, s3: 1024, s4: 2048
        resnet_s1_dim = 256  # Output channels of ResNet Layer1
        # self.adapter_s1 = VisionAdapter(resnet_s1_dim    , self.vision_dim, pool_kernel=8, pool_stride=8) # 56x56 -> 7x7
        # self.adapter_s2 = VisionAdapter(resnet_s1_dim * 2, self.vision_dim, pool_kernel=4, pool_stride=4) # 28x28 -> 7x7
        # self.adapter_s3 = VisionAdapter(resnet_s1_dim * 4, self.vision_dim, pool_kernel=2, pool_stride=2) # 14x14 -> 7x7
        # self.adapter_s4 = VisionAdapterV2(resnet_s1_dim * 8, self.vision_dim, pool_kernel=1, pool_stride=1) # 7x7 -> 7x7
        
        self.adapter_s4 = LinearBlock(resnet_s1_dim * 8, resnet_s1_dim * 2)
        
        # self.adapter_s1 = MonaRes(resnet_s1_dim, inter_dim=128, out_dim=256, pool=(8, 8), hw_shapes=(56, 56))
        # self.adapter_s2 = MonaRes(resnet_s1_dim * 2, inter_dim=128, out_dim=256, pool=(4, 4), hw_shapes=(28, 28))
        # self.adapter_s3 = MonaRes(resnet_s1_dim * 4, inter_dim=128, out_dim=256, pool=(2, 2), hw_shapes=(14, 14))
        # self.adapter_s4 = MonaRes(resnet_s1_dim * 8, inter_dim=128, out_dim=256, pool=(1, 1), hw_shapes=(7, 7))
        
        # Combiner for adapted multi-scale features
        # 这个方式是最好的吗？
        # self.multi_scale_combiner = MultiScaleCombiner(2 * self.vision_dim, self.vision_dim // 2, config.drop_rate)
        
        # Cross Attention
        # (seq_len, batch, embed_dim)
        # self.cross_attention = CrossAttention(embed_dim=self.vision_dim, num_heads=8, layers=2)
        
        # AU-specific linear layers
        # self.au_specific_linears = nn.ModuleList([
        #     nn.Sequential(nn.Conv2d(self.vision_dim, self.vision_dim, kernel_size=1, bias=False),
        #                   nn.BatchNorm2d(self.vision_dim),
        #                   nn.ReLU(inplace=True),
        #                   nn.Dropout(config.drop_rate))
        #     for _ in range(self.num_classes)
        # ])
        
        au_specific_linears = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.vision_dim, self.vision_dim)
            au_specific_linears += [layer]
        
        self.au_specific_linears = nn.ModuleList(au_specific_linears)
        
        
        # --- Graph Network ---
        self.gnn = GNN(self.vision_dim, num_classes=self.num_classes, neighbor_num=config.neighbor_num, metric='cosine')

        # --- Independent AU Classifiers ---
        # Create a separate linear classifier for each AU
        self.au_classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.vision_dim, 1))
            for _ in range(self.num_classes)
        ])
        self.sigmoid = nn.Sigmoid() # To convert logits to probabilities
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.class_num),
        )
        # # 迁移学习loss
        # bottleneck_layer_list = [
        #     nn.Linear(1024, self.config.bottleneck_width),
        #     nn.BatchNorm1d(self.config.bottleneck_width),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # ]
        # self.bottleneck_layer = nn.Sequential(*bottleneck_layer_list)
        # self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        # self.bottleneck_layer[0].bias.data.fill_(0.1)
        
        # 注册钩子函数，捕获 ResNet50 的 4 个 stack 的输出
        self.vision_features = {}
        self.image_encoder.layer1.register_forward_hook(self.get_activation('layer1'))
        self.image_encoder.layer2.register_forward_hook(self.get_activation('layer2'))
        self.image_encoder.layer3.register_forward_hook(self.get_activation('layer3'))
        self.image_encoder.layer4.register_forward_hook(self.get_activation('layer4'))
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.vision_features[name] = output
        return hook
    
    def freeze_bn(self):
        for module in self.image_encoder.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
    
    def forward(self, image, text=None, labels=None):

        B, C, H, W = image.shape # images: [128, 3, 224, 224]
        
        # 1. text features
        # prompts = self.prompt_learner() # [10, 77, 512]
        # tokenized_prompts = self.tokenized_prompts # [10, 77]
        # text_features = self.text_encoder(prompts, tokenized_prompts) # [10, 1024]
        # text_features = self.text_transform(text_features) # [10, 256]
        
        # 2. image features
        ## 2.1 Extract Multi-Scale Vision Features via Hooks
        image_features = self.image_encoder(image.type(self.dtype)) # RN50: [128, 1024]
        # features_s1 = self.vision_features['layer1'] # Shape: [B, 256, 56, 56]
        # features_s2 = self.vision_features['layer2'] # Shape: [B, 512, 28, 28]
        # features_s3 = self.vision_features['layer3'] # Shape: [B, 1024, 14, 14]
        features_s4 = self.vision_features['layer4'] # Shape: [B, 2048, 7, 7]
        ## 2.2 Adapt and Pool Features
        # adapted_s1 = self.adapter_s1(features_s1) # Shape: [B, 256, 7, 7]
        # adapted_s2 = self.adapter_s2(features_s2) # Shape: [B, 256, 7, 7]
        # adapted_s3 = self.adapter_s3(features_s3) # Shape: [B, 256, 7, 7]
        adapted_s4 = self.adapter_s4(features_s4) # Shape: [B, 256, 7, 7]
        ## 2.3 Combine Multi-Scale Features
        # combined_features = torch.cat([adapted_s1, adapted_s2, adapted_s3, adapted_s4], dim=1) # Shape: [B, 1024, 7, 7]
        # combined_features = self.multi_scale_combiner(combined_features) # Shape: [B, 256, 7, 7]
        
        # # 3. cross attention
        # # combined_features
        # B, cross_dim, H, W = combined_features.shape
        # combined_features_cross = combined_features.reshape(B, cross_dim, H*W).permute(2, 0, 1) # [49, 256, 256]
        # text_features_cross = text_features.unsqueeze(1).expand(-1, B, -1) # [10, B, 256]
        # # (seq_len, batch, embed_dim) [49, 256, 1024]
        # vision_features_cross = self.cross_attention(combined_features_cross, text_features_cross, text_features_cross)
        # vision_features_cross = vision_features_cross.permute(1, 2, 0).reshape(B, cross_dim, H, W)
        # vision_features_cross = vision_features_cross + combined_features # [B, 256, 7, 7]
        
        vision_features_cross = adapted_s4
        
        # 4. Apply AU-specific Layers and Pool
        au_features = []
        for layer in self.au_specific_linears:
            au_features.append(layer(vision_features_cross).unsqueeze(1)) # Shape: 10 * [B, 1, 256, 7, 7]
        au_features = torch.cat(au_features, dim=1) # Shape: [B, 10, 256, 7, 7]
        au_pooled_features = au_features.reshape(B, self.config.class_num, 512, -1) # [B, 10, 256, 49]
        au_pooled_features = au_pooled_features.mean(dim=-1) # Shape: [B, num_classes, 512]
        
        # 5. Graph Neural Network Processing
        # pdb.set_trace()
        gnn_features = self.gnn(au_pooled_features) # Shape: [B, num_classes, 256]

        # 6. Independent Classification for each AU
        au_logits = []
        for i in range(self.num_classes):
            au_feature = gnn_features[:, i, :] # Shape: [B, vision_dim]
            au_feature = F.normalize(au_feature, p=2, dim=-1)
            logit = self.au_classifiers[i](au_feature) # Shape: [B, 1]
            au_logits.append(logit)
        au_logits = torch.cat(au_logits, dim=1) # Shape: [B, num_classes]
        
        return {
            'logits': au_logits,
            'au_vision_features': gnn_features, # [B, num_classes, 256]
            # 'au_text_features': text_features,  # [num_classes, 256]
        }

if __name__ == "__main__":
    model = TransferNet()

    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"The output of model is: {output}")
    # print(f"Model: {model}")