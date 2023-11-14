import torch
import torch.nn as nn
from torch import Tensor
from pickle import load
import os.path as osp
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Glove_Transformer(nn.Module):
    def __init__(
        self, 
        embed_dim: int = 200,
        drop_rate: float = 0.5,
        text_features: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dataset_dir: str = 'data/flickr8k',
    ) -> None:
        super().__init__()
        
        self.embed = nn.Embedding.from_pretrained(
            self.load_weight_embedding(dataset_dir),
            freeze=True,
            padding_idx=0)
        
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)
        
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=drop_rate,
        )
        self.linear_image = nn.Linear(text_features, embed_dim)
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
    
    def forward(self, image: Tensor, sequence: Tensor) -> Tensor:
        # from IPython import embed; embed()
        tgt = self.embed(sequence)
        tgt = self.pos_encoder(tgt)
        
        src = self.linear_image(image)
        src = torch.unsqueeze(src, 0).permute(1, 0, 2)
        src = src.repeat(1, tgt.shape[1], 1)
        
        out = self.transformer(src, tgt)
        out = self.linear(out)
        # return out[:, -1]  # only get the last
        return out
if __name__ == "__main__":
    net = Glove_Transformer()

    sequence = torch.randint(0, 100, (2, 20))
    out = net(sequence)
    
    print(out.shape)