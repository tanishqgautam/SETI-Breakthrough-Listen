import timm
import torch.nn as nn

from .config import Config


class SwinNet(nn.Module):
    def __init__(self, model_name= Config.model, out_features=Config.out_features,
                 inp_channels=Config.inp_channels, pretrained=True):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       in_chans=inp_channels)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, out_features, bias=True)    
    
    def forward(self, x):
        x = self.model(x)
        return x