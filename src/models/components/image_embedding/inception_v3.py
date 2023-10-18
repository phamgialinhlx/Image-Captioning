import copy
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.inception import inception_v3, InceptionOutputs


class InceptionNet(nn.Module):

    def __init__(self,
                 image_features: int = 256,
                 device: str = 'cpu',
                 drop_rate: float = 0.5) -> None:
        super().__init__()

        self.inception_encoder = inception_v3(weights='DEFAULT',
                                              transform_input=False).to(device)
        self.inception_encoder.fc = torch.nn.Identity()
        self.inception_encoder.requires_grad_(False)

        # for param in self.inception_encoder.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(p=drop_rate)
        self.linear = nn.Linear(2048, image_features)
        self.relu = nn.ReLU()

    def forward(self, images: Tensor) -> Tensor:
        ### output: batch, 256
        with torch.no_grad():
            # self.inception_encoder.eval()
            # if in train mode, inception have batch_normalization so batch_size > 1

            embed = self.inception_encoder(images)
            if isinstance(embed, InceptionOutputs):
                embed = embed[0]

        embed = self.dropout(embed)
        embed = self.relu(self.linear(embed))
        return embed


if __name__ == "__main__":
    net = InceptionNet(device='cpu')

    x = torch.randn(2, 3, 299, 299)
    out = net(x)
    print(out.shape)