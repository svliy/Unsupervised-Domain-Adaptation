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
from .losses import WeightedAsymmetricLoss, ContrastiveLossInfoNCE, MMD_loss

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
        x = self.drop(x) # [B, 49, 512]
        x = self.fc(x).permute(0, 2, 1) # [B, 512, 49]
        x = self.relu(self.bn(x)).permute(0, 2, 1) # [B, 49, 512]
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
        self.conv1 = nn.Conv2d(4 * self.vision_dim, bottle_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1  = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(bottle_dim, 4 * self.vision_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 *self.vision_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2  = nn.Dropout(dropout)
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

        out = x + out
        out = self.relu2(out)
        out = self.drop2(out)
        return out

class TransferNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.class_num
        self.vision_dim = 256
        
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
        
        print("冻结CLIP模型视觉和文本编码器")
        for param in list(self.image_encoder.parameters()) + list(self.text_encoder.parameters()):
            param.requires_grad = False
        self.clip_logit_scale.requires_grad = False
        
        # Vision Adapters for multi-scale features
        # Assuming ResNet50 stages output: s1: 256, s2: 512, s3: 1024, s4: 2048
        # Adapting all to self.vision_dim (512)
        resnet_s1_dim = 256  # Output channels of ResNet Layer1
        self.adapter_s1 = VisionAdapter(resnet_s1_dim    , self.vision_dim, pool_kernel=8, pool_stride=8) # 56x56 -> 7x7
        self.adapter_s2 = VisionAdapter(resnet_s1_dim * 2, self.vision_dim, pool_kernel=4, pool_stride=4) # 28x28 -> 7x7
        self.adapter_s3 = VisionAdapter(resnet_s1_dim * 4, self.vision_dim, pool_kernel=2, pool_stride=2) # 14x14 -> 7x7
        self.adapter_s4 = VisionAdapter(resnet_s1_dim * 8, self.vision_dim, pool_kernel=1, pool_stride=1) # 7x7 -> 7x7
        
        # Combiner for adapted multi-scale features
        # Input dim is 4 * self.vision_dim because we concat 4 feature maps of vision_dim channels
        # 这个方式是最好的吗？
        self.multi_scale_combiner = MultiScaleCombiner(self.vision_dim, self.vision_dim // 2, config.drop_rate)
        
        # AU-specific linear layers
        # self.au_specific_linears = nn.ModuleList([
        #     nn.Sequential(MultiScaleCombiner(self.vision_dim, self.vision_dim // 2, dropout=config.drop_rate),
        #                   MultiScaleCombiner(self.vision_dim, self.vision_dim // 2, dropout=config.drop_rate)) 
        #     for _ in range(self.num_classes)
        # ])
        
        # --- Graph Network ---
        # self.gnn = GNN(4*self.vision_dim, num_classes=self.num_classes, neighbor_num=self.config.neighbor_num, metric='cosine')

        # --- Independent AU Classifiers ---
        # Create a separate linear classifier for each AU
        # self.au_classifiers = nn.ModuleList([
        #     nn.Linear(4*self.vision_dim, 1) for _ in range(self.num_classes)
        # ])
        self.sigmoid = nn.Sigmoid() # To convert logits to probabilities
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, config.class_num),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 10)
        )
        # 迁移学习loss
        bottleneck_layer_list = [
            nn.Linear(1024, self.config.bottleneck_width),
            nn.BatchNorm1d(self.config.bottleneck_width),
            nn.ReLU(),
            nn.Dropout(0.5)
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_layer_list)
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        
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
    
    def forward(self, image, text=None):

        B, C, H, W = image.shape # images: [128, 3, 224, 224]
        
        # 1. text_features
        # prompts = self.prompt_learner()
        # tokenized_prompts = self.tokenized_prompts
        # text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # 2. image_features
        ## 2.1 Extract Multi-Scale Vision Features via Hooks
        image_features = self.image_encoder(image.type(self.dtype)) # RN50: [128, 1024]
        features_s1 = self.vision_features['layer1'] # Shape: [B, 256, 56, 56]
        features_s2 = self.vision_features['layer2'] # Shape: [B, 512, 28, 28]
        features_s3 = self.vision_features['layer3'] # Shape: [B, 1024, 14, 14]
        features_s4 = self.vision_features['layer4'] # Shape: [B, 2048, 7, 7]
        
        ## 2.2 Adapt and Pool Features
        adapted_s1 = self.adapter_s1(features_s1) # Shape: [B, 256, 7, 7]
        adapted_s2 = self.adapter_s2(features_s2) # Shape: [B, 256, 7, 7]
        adapted_s3 = self.adapter_s3(features_s3) # Shape: [B, 256, 7, 7]
        adapted_s4 = self.adapter_s4(features_s4) # Shape: [B, 256, 7, 7]
        ## 2.3 Combine Multi-Scale Features
        combined_features = torch.cat([adapted_s1, adapted_s2, adapted_s3, adapted_s4], dim=1) # Shape: [B, 1024, 7, 7]
        
        combined_features = self.multi_scale_combiner(combined_features) # Shape: [B, 1024, 7, 7]
        
        combined_features_avg = self.global_avg_pool(combined_features)
        combined_features_avg = torch.flatten(combined_features_avg, 1)
        combined_features_avg = self.bottleneck_layer(combined_features_avg)
        
        # Reshape for sequence processing: (B, C, H, W) -> (B, H*W, C)
        # combined_features_seq = combined_features.view(B, 4*self.vision_dim, -1).permute(0, 2, 1) # Shape: [B, 49, 1024]
        
        # 3. Apply AU-specific Layers and Pool
        # au_raw_features = []
        # for layer in self.au_specific_linears:
        #     au_raw_features.append(layer(combined_features).unsqueeze(1)) # Shape: [B, 1024, 7, 7]
            
        # au_raw_features = torch.cat(au_raw_features, dim=1) # Shape: [B, , 49, 512]
        # au_pooled_features = au_raw_features.mean(dim=-2) # Shape: [B, num_classes, 512]
        
        # # 4. Graph Neural Network Processing
        # gnn_features = self.gnn(au_pooled_features) # Shape: [B, num_classes, 512]
        
        # # 5. Independent Classification for each AU
        # au_logits = []
        # for i in range(self.num_classes):
        #     # Extract the feature vector for the i-th AU for all images in the batch
        #     au_feature = gnn_features[:, i, :] # Shape: [B, vision_dim]
        #     # Apply the i-th independent classifier
        #     logit = self.au_classifiers[i](au_feature) # Shape: [B, 1]
        #     au_logits.append(logit)
        # # Concatenate logits from all AU classifiers
        # # [128, 10]
        # au_logits = torch.cat(au_logits, dim=1) # Shape: [B, num_classes]
        
        # # 6. Apply Sigmoid to get probabilities
        # au_probabilities = self.sigmoid(au_logits) # Shape: [B, num_classes]
        
        # 测试代码
        # combined_features Shape: [B, 1024, 7, 7]
        # combined_features = combined_features.reshape(B, 1024, 49)
        # combined_features = combined_features.mean(dim=-1)
        logits = self.head(combined_features) # [B, class_num]
        
        
        
        return {
            'logits': logits,
            'transfer_feature': combined_features_avg,
            # 'probabilities': au_probabilities,
            # 'vision_features': gnn_features,
            # 'text_features': text_features, # [10, 1024]
        }

if __name__ == "__main__":
    model = TransferNet()

    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"The output of model is: {output}")
    # print(f"Model: {model}")