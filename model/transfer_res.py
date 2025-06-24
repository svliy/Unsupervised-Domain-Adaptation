import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import pdb

class TransferRes(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        if self.config.backbone == 'ori_RN50':
            weights = ResNet50_Weights.DEFAULT
            self.base_model = resnet50(weights=weights)
        
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, self.config.class_num)
        

    def forward(self, x):
        x = self.base_model(x)
        return {
            'logits': x
        }
    
    def freeze_bn(self):
        pass
    
if __name__ == '__main__':
    
    class Configuration():
        def __init__(self):
            self.backbone = 'ori_RN50'
            self.class_num = 5
    
    config = Configuration()
    
    model = TransferRes(config=config)
    
    data_input = torch.randn(1, 3, 224, 224)
    data_output = model(data_input)
    print(f"The shape of output: {data_output.shape}")
    print(model)