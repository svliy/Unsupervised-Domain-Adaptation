import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from .clip import clip
from .graph import GNN
from .description import describe_au
from .transformers_encoder.transformer import TransformerEncoder as CrossAttention
from .losses import *

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
    def __init__(self, config, transfer_loss='mmd') -> None:
        super().__init__()
        
        
        self.config = config
        self.class_num = config.class_num
        self.transfer_loss = transfer_loss
        
        clip_model, preprocess = clip.load(config.backbone, device="cpu")
        self.vision_encoder = clip_model.visual
        self.vision_dim = 512 # 临时
        
        self.avgpool_global = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(2 * self.vision_dim, self.vision_dim // 2),
            nn.BatchNorm1d(self.vision_dim // 2), # 为什么是1D
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.head = nn.Sequential(
            nn.Linear(2 * self.vision_dim, self.vision_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.vision_dim, self.class_num)
        )
   
    def forward(self, source, target):
        # source: [64, 3, 224, 224]
        
        # 获取多尺度图像特征
        source_f_v = self.vision_encoder(source) # [B, 1024]
        target_f_v = self.vision_encoder(target) # [B, 1024]
        
        source_clf = self.head(source_f_v) # [B, 10]
        
        source_f_v = self.bottleneck_layer(source_f_v) # [B, 256]
        target_f_v = self.bottleneck_layer(target_f_v) # [B, 256]
        # pdb.set_trace()
        transfer_loss = self.adapt_loss(source_f_v, target_f_v, self.transfer_loss)

        return {
            'source_clf': source_clf,
            'transfer_loss': transfer_loss,
        }
        
    def predict(self, x):
        features = self.vision_encoder(x)
        clf = self.head(features)
        return clf
    
    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            # Your own loss
            loss = 0
        return loss

if __name__ == "__main__":
    model = TransferAU()

    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"The output of model is: {output}")
    # print(f"Model: {model}")