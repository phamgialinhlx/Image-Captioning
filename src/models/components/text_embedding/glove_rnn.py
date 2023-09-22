import torch
import torch.nn as nn
from torch import Tensor

class Glove_RNN(nn.Module):

    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int,
                 hidden_size: int,
                 n_layer_rnn: int = 1,
                 ) -> None:
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embed_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=hidden_size,
                          num_layers=n_layer_rnn,
                          batch_first=False)
        

    def forward(self, sequence: Tensor) -> Tensor:
        out = self.embed(sequence)
        out = self.dropout(out)
        out, _ = self.rnn(out) # return output and hidden state
        return out[-1] # only get 

if __name__ == "__main__":
    net = Glove_RNN(vocab_size=2005, 
                    embed_dim=128,
                    hidden_size=256,
                    n_layer_rnn=2)
    
    x = torch.randint(0, 100, (20, 2))
    out = net(x)
    print(out.shape)