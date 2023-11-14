import torch
import torch.nn as nn
from torch import Tensor
from pickle import load
import os.path as osp
from pytorch_pretrained_bert import BertTokenizer, BertModel

class Bert(nn.Module):
    def __init__(
       self,
       drop_rate: float = 0.5,
       text_features: int = 256,
    ) -> None:
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=drop_rate)
        self.linear = nn.Linear(self.model.config.hidden_size, text_features)

    def forward(self, sequence: Tensor) -> Tensor:
       sequence = self.tokenizer.tokenize(sequence)
       sequence = self.model(**sequence)
       sequence = sequence.last_hidden_state[:, 0, :]
       sequence = self.dropout(sequence)
       sequence = self.linear(sequence)
       return sequence

if __name__ == "__main__":
   net = Bert()

   x = torch.randint(0, 100, (2, 20))
   out = net(x)
   print(x.shape, out.shape)
