import torch.nn as nn
from torch import Tensor

class InceptionNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, img: Tensor) -> Tensor:
        ### output: batch, 256
        pass

