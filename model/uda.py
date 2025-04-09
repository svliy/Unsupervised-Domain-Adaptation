import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from .clip import clip

class TransferAU(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.class_num = config.class_num
        
        clip_model, preprocess = clip.load(config.backbone, device="cpu")
        self.vision_encoder = clip_model.visual
        self.vision_dim = 512 # 临时
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
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * self.vision_dim, 256),
            nn.Linear(256, 256),
            nn.Linear(256, self.class_num),
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
    
    def forward(self, x):
        # x: [64, 3, 224, 224]
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
        
        # 特征融合
        v_f_fusion = torch.cat((v_f_s1_same, v_f_s2_same, v_f_s3_same, v_f_s4_same), dim=1) # [128, 2048, 7, 7]
        v_f_fusion = self.avgpool_global(v_f_fusion) # [128, 2048, 1, 1]
        
        output = self.head(v_f_fusion) # [128, 10]
        output = self.sigmoid(output)

        # pdb.set_trace()
        return output

if __name__ == "__main__":
    model = TransferAU()

    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"The output of model is: {output}")
    # print(f"Model: {model}")