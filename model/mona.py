import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Mona(nn.Module):
    # [B, L, C]
    # [256, 197, 768]
    def __init__(self, in_dim, inter_dim=64, factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, inter_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(inter_dim, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(inter_dim)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2
    
class MonaRes(nn.Module):
    # [B, L, C]
    # [256, 197, 768]
    def __init__(self, in_dim, inter_dim=64, factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, inter_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(inter_dim, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(inter_dim)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1) # [B, H*W, C]
        
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        
        out = identity + project2
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out
    
if __name__ == '__main__':
    
    model = MonaRes(256)
    print(f"Mona: {model}")
    print(f"模型总参数量: {sum(param.numel() for param in model.parameters())}")
    
    
    # [B, L, C]
    # input = torch.randn(10, 49, 1024)
    # [B, C, H, W]
    input = torch.randn(3, 256, 56, 56)
    print(f"The input size is: {input.shape}")
    output = model(input, hw_shapes=(56, 56))
    print(f"The output size is: {output.shape}")
    # for param in model.parameters():
    #     print(type(param))
    #     print(param.shape)
        
        