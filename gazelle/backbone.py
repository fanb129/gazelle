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

        if "vitl" in model_name:
            self.out_indices = [4, 11, 17, 23] 
        elif "vitb" in model_name:
            self.out_indices = [2, 5, 8, 11]
        else:
            self.out_indices = [len(self.model.blocks) - 1]

    def forward(self, x):
        features = self.model.get_intermediate_layers(
            x,
            n=self.out_indices, 
            reshape=True
        )
        # features 是一个 list，包含 4 个 tensor，每个形状为 [B, C, H, W]
        return features
    
    def get_dimension(self):
        return self.model.embed_dim
    
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