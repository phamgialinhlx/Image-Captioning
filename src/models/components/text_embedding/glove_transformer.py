import torch
import torch.nn as nn
from torch import Tensor
from pickle import load
import os.path as osp

class Glove_Transfomer(nn.Module):
    def __init__(
        self, 
        embed_dim: int = 200,
        drop_rate: float = 0.5,
        text_features: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        dataset_dir: str = 'data/flickr8k',
    ) -> None:
        super().__init__()
        
        self.embed = nn.Embedding.from_pretrained(
            self.load_weight_embedding(dataset_dir),
            freeze=True,
            padding_idx=0)

        self.encoder = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=0,
            dropout=drop_rate,
        ).encoder
                
        self.linear = nn.Linear(embed_dim, text_features)
        
    def load_weight_embedding(self, dataset_dir: str = 'data/flickr8k'):
        embedding_matrix_path = osp.join(dataset_dir, 'embedding_matrix.pkl')

        if not osp.exists(embedding_matrix_path):
            raise ValueError(
                "weight_embedding_path is not exist. Please check path or run datamodule to prepare"
            )

        with open(embedding_matrix_path, "rb") as file:
            embedding_matrix = load(file)
        print('Embedding_matrix:', embedding_matrix.shape)
        return embedding_matrix
    def forward(self, sequence: Tensor) -> Tensor:
       out = self.embed(sequence)
       out = self.encoder(out)
       out = self.linear(out)
    #    out = self.relu(out)
       return out[:, -1]  # only get the last

if __name__ == "__main__":
    net = Glove_Transfomer()
    print(net.encoder)

    sequence = torch.randint(0, 100, (2, 20))
    out = net(sequence)
    
    print(out.shape)