from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Abstract Backbone class
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self):
        pass


class DinoV3Backbone(Backbone):
    def __init__(self, model_name):
        super(DinoV3Backbone, self).__init__()
        self.model = torch.hub.load('dinov3', model_name, source='local', pretrained=False)
        self.model.load_state_dict(torch.load('./checkpoints/'+model_name+"_pretrain.pth"))

        # 2. 确定要提取的层数 (参考 DINOv3 官方配置)
        if "vitl" in model_name:
            self.out_indices = [4, 11, 17, 23] # 官方推荐的 ViT-L 用于密集任务的层
        elif "vitb" in model_name:
            self.out_indices = [2, 5, 8, 11] # 示例，可以取最后4层或者均匀分布
        else:
            self.out_indices = [len(self.model.blocks) - 1] # 默认只取最后一层

    def forward(self, x):
        # b, c, h, w = x.shape
        # out_h, out_w = self.get_out_size((h, w))
        # x = self.model.forward_features(x)['x_norm_patchtokens']
        # x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2) # "b (out_h out_w) c -> b c out_h out_w"
        # return x
        # 3. 使用 get_intermediate_layers 获取多层特征
        # reshape=True 会自动帮你把 (B, N, C) 变为 (B, C, H, W)，且自动处理掉 register tokens
        features = self.model.get_intermediate_layers(
            x,
            n=self.out_indices, 
            reshape=True
        )
        
        # 4. 特征融合策略
        # 策略A (简单): 拼接所有层 (通道数变大，例如 1024*4)
        # 结果形状: (B, C * len(indices), H, W)
        return torch.cat(features, dim=1) 
        
        # 策略B (常用): 只返回最后一层 (如果你不想改下游网络结构)
        # return features[-1]

        # 策略C (推荐): 返回最后一层，但建议你尝试策略A并在下游加一个 1x1 卷积降维
        # return features[-1]
    
    def get_dimension(self):
        # return self.model.embed_dim
        # 【关键修改】返回拼接后的总通道数
        # 例如 ViT-L: 1024 * 4 = 4096
        return self.model.embed_dim * len(self.out_indices)
    
    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.patch_size, w // self.model.patch_size)
    
    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            transforms.Resize(in_size),
        ])