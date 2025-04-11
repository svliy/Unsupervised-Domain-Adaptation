import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from .clip import clip
from .graph import GNN
from .description import describe_au
from .transformers_encoder.transformer import TransformerEncoder as CrossAttention

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

class TransferAU(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.class_num = config.class_num
        
        self.config = config
        
        clip_model, preprocess = clip.load(config.backbone, device="cpu")
        self.vision_encoder = clip_model.visual
        self.vision_dim = 512 # 临时
        self.vision_transform = nn.Linear(4 * self.vision_dim, self.vision_dim) # LLM模型中如何对齐两个模态的维度

        # 文本
        self.text_encoder = TextEncoder(clip_model)
        self.text_transform = nn.Linear(1024, self.vision_dim) # LLM模型中如何对齐两个模态的维度
        
        # self.logit_scale = clip_model.logit_scales
        self.avgpool_global = nn.AdaptiveAvgPool2d((1, 1))

        # 适配器1
        self.vision_adapter_s1 = nn.Sequential(
            nn.Conv2d(self.vision_dim // 2, self.vision_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.vision_dim),
            nn.ReLU(inplace=True)
        )
        # 适配器2
        self.vision_adapter_s2 = nn.Sequential(
            nn.Conv2d(self.vision_dim, self.vision_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.vision_dim),
            nn.ReLU(inplace=True)
        )
        # 适配器3
        self.vision_adapter_s3 = nn.Sequential(
            nn.Conv2d(2 * self.vision_dim, self.vision_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.vision_dim),
            nn.ReLU(inplace=True)
        )
        # 适配器4
        self.vision_adapter_s4 = nn.Sequential(
            nn.Conv2d(4 * self.vision_dim, self.vision_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.vision_dim),
            nn.ReLU(inplace=True)
        )
        self.vision_adapter_s1_pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.vision_adapter_s2_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.vision_adapter_s3_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.vision_adapter_s4_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        
        # cross attention
        self.cross_attention = CrossAttention(embed_dim=self.vision_dim, num_heads=8, layers=2)
        
        # 定义图神经网络
        self.gnn = GNN(self.vision_dim, num_classes=4, neighbor_num=4, metric='cosine')
        
        self.head = nn.Sequential(
            nn.Linear(self.vision_dim, self.class_num),
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # 注册钩子函数，捕获 ResNet50 的 4 个 stack 的输出
        self.vision_features = {}
        clip_model.visual.layer1.register_forward_hook(self.get_activation('layer1'))
        clip_model.visual.layer2.register_forward_hook(self.get_activation('layer2'))
        clip_model.visual.layer3.register_forward_hook(self.get_activation('layer3'))
        clip_model.visual.layer4.register_forward_hook(self.get_activation('layer4'))
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.vision_features[name] = output
        return hook
    
    def forward(self, x, texts_ori=None):
        # x: [64, 3, 224, 224]
        B, C, H, W = x.shape
        # 获取多尺度图像特征
        image_features = self.vision_encoder(x) # RN50: [128, 1024]
        v_f_s1 = self.vision_features['layer1'] # [128, 256, 56, 56]
        v_f_s2 = self.vision_features['layer2'] # [128, 512, 28, 28]
        v_f_s3 = self.vision_features['layer3'] # [128, 1024, 14, 14]
        v_f_s4 = self.vision_features['layer4'] # [128, 2048, 7, 7]
        
        # 特征变换
        v_f_s1_same = self.vision_adapter_s1(v_f_s1) # [128, 512, 56, 56]
        v_f_s1_same = self.vision_adapter_s1_pool(v_f_s1_same) # [128, 512, 7, 7]
        
        v_f_s2_same = self.vision_adapter_s2(v_f_s2) # [128, 512, 28, 28]
        v_f_s2_same = self.vision_adapter_s2_pool(v_f_s2_same) # [128, 512, 7, 7]
        
        v_f_s3_same = self.vision_adapter_s3(v_f_s3) # [128, 512, 14, 14]
        v_f_s3_same = self.vision_adapter_s3_pool(v_f_s3_same) # [128, 512, 7, 7]
        
        v_f_s4_same = self.vision_adapter_s4(v_f_s4) # [128, 512, 7, 7]
        v_f_s4_same = self.vision_adapter_s4_pool(v_f_s4_same) # [128, 512, 7, 7]
        
        # 跨模态注意力
        v_f_sets = torch.stack([v_f_s1_same, v_f_s2_same, v_f_s3_same, v_f_s4_same]) # [4, 128, 512, 7, 7]
        v_f_sets = v_f_sets.permute(1, 0, 2, 3, 4).contiguous().view(B*4, self.vision_dim, -1) # [128*4, 512, 49]
        v_f_sets = v_f_sets.permute(2, 0, 1) # [49, 128*4, 512]
        
        # 获取文本特征
        texts_ori_ids = clip.tokenize(texts_ori).cuda()
        texts_ori_f = self.text_encoder(texts_ori_ids) # [10, 1024]
        texts_ori_f = self.text_transform(texts_ori_f) # [10, 512]
        texts_ori_f_cross = texts_ori_f.unsqueeze(0).repeat(B*4, 1, 1) # [128*4, 10, 512]
        texts_ori_f_cross = texts_ori_f_cross.permute(1, 0, 2).contiguous() # [10, 128*4, 512]
        inter_f = self.cross_attention(v_f_sets, texts_ori_f_cross, texts_ori_f_cross) # [49, 128*4, 512]
        # pdb.set_trace()
        inter_f = inter_f + v_f_sets # [49, 128*4, 512]
        
        # 构造图神经网络
        g_nodes = inter_f.permute(1, 0, 2).contiguous().view(B, 4, -1, self.vision_dim) # [128, 4, 49, 512]
        g_nodes = g_nodes.mean(dim=-2) # [128, 4, 512]
        g_nodes = self.gnn(g_nodes) # [128, 4, 512]
        
        # g_nodes = torch.stack([v_f_s1_same, v_f_s2_same, v_f_s3_same, v_f_s4_same]) # [4, 128, 512, 7, 7]
        # g_nodes = g_nodes.permute(1, 0, 2, 3, 4).contiguous().view(B, 4, self.vision_dim, -1) # [128, 4, 512, 49]
        # g_nodes = g_nodes.mean(dim=-1) # [128, 4, 512]
        # g_nodes = self.gnn(g_nodes) # [128, 4, 512]
        
        v_f_fusion = torch.flatten(g_nodes, start_dim=1) # [128, 2048]
        # v_f_fusion = self.avgpool_global(v_f_fusion) # [128, 2048, 1, 1]
        # pdb.set_trace()
        v_f_final = self.vision_transform(v_f_fusion)
        output = self.head(v_f_final) # [128, 10]
        # output = self.sigmoid(output)

        # pdb.set_trace()
        return {
            'logits': output,
            'vision_features': v_f_final,
            "text_features": texts_ori_f,
        }

if __name__ == "__main__":
    model = TransferAU()

    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"The output of model is: {output}")
    # print(f"Model: {model}")