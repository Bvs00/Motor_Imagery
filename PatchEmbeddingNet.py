import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
import math

################ PATCHEMBEDDING ######################
class PatchEmbeddingNet(nn.Module):
    def __init__(self, f1=8, kernel_size=63, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, channels=3, num_classes=2, model_name_prefix="PatchEmbeddingNet", samples=1000):
        super().__init__()
        f2 = D*f1
        self.channels = channels
        self.samples = samples
        self.model_name_prefix = model_name_prefix
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (channels, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 15), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.classification = nn.Linear(self.num_features_linear(), num_classes)
    
    def num_features_linear(self):
        x = torch.ones((1, 1, self.channels, self.samples))
        x = self.cnn_module(x)
        x = self.projection(x)
        return x.shape[-1]*x.shape[-2]
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        x = x.flatten(start_dim=1)
        x = self.classification(x)
        return x
