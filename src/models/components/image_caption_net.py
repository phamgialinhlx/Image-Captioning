import torch
import rootutils
import os.path as osp
import torch.nn as nn
from torch import Tensor
from pickle import load
from torch.nn.utils.rnn import pad_sequence

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.image_embedding import InceptionNet
from src.models.components.text_embedding import Glove_RNN
from src.models.components.attention import Attention


class ImageCaptionNet(nn.Module):

    def __init__(self,
                 image_embed_net,
                 text_embed_net,
                 features: int = 256,
                 dataset_dir: str = 'data/flickr8k',
                 operation: str = 'add') -> None:
        """_summary_

        Args:
            image_embed_net (_type_): _description_
            text_embed_net (_type_): _description_
            features (int, optional): _description_. Defaults to 256.
            dataset_dir (str, optional): _description_. Defaults to 'data/flickr8k'.
        """
        super().__init__()

        self.text_embed_net = text_embed_net
        self.image_embed_net = image_embed_net

        self.id2word, self.word2id, self.max_length, vocab_size = self.prepare(
            dataset_dir)

        self.operation = operation
        if self.operation == 'concat':
            self.linear = nn.Linear(features << 1, features)
        elif self.operation == 'cross_attention':
            self.cross_att = Attention(channels=features)
        elif self.operation == 'self_attention':
            self.linear = nn.Linear(features << 1, features)
            self.self_att = Attention(channels=features)
        elif self.operation != 'add':
            raise NotImplementedError(f"unknown operation: {self.operation}")

        self.linear_1 = nn.Linear(features, features)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(features, vocab_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, image: Tensor, sequence: Tensor) -> Tensor:
        """_summary_

        Args:
            image (Tensor): (batch, c, w, h)
            sequence (Tensor): (batch, max_length)

        Returns:
            Tensor: (batch, vocab_size)
        """
        image_embed = self.image_embed_net(image)
        sequence_embed = self.text_embed_net(sequence)

        # integrate two embedding vector
        if self.operation == 'add':
            embed = image_embed + sequence_embed
        elif self.operation == 'concat':
            embed = torch.cat((image_embed, sequence_embed), dim=1)
            embed = self.linear(embed)
        elif self.operation == 'cross_attention':
            embed = self.cross_att(image_embed, sequence_embed)
        elif self.operation == 'self_attention':
            embed = torch.cat((image_embed, sequence_embed), dim=1)
            embed = self.linear(embed)
            embed = self.self_att(embed)
        elif self.operation != 'add':
            raise NotImplementedError(f"unknown operation: {self.operation}")

        out = self.relu(self.linear_1(embed))
        # out = self.softmax(self.linear_2(out))
        out = self.linear_2(out)
        return out

    def prepare(self, dataset_dir: str):
        """_summary_

        Args:
            dataset_dir (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        id2word_path = osp.join(dataset_dir, 'id2word.pkl')
        word2id_path = osp.join(dataset_dir, 'word2id.pkl')
        max_length_path = osp.join(dataset_dir, 'max_length.pkl')
        vocab_size_path = osp.join(dataset_dir, 'vocab_size.pkl')

        if not osp.exists(vocab_size_path):
            raise ValueError(
                "vocab_size_path is not exist. Please check path or run datamodule to prepare"
            )

        with open(id2word_path, "rb") as file:
            id2word = load(file)

        with open(word2id_path, "rb") as file:
            word2id = load(file)

        with open(max_length_path, "rb") as file:
            max_length = load(file)

        with open(vocab_size_path, "rb") as file:
            vocab_size = load(file)

        return id2word, word2id, max_length, vocab_size

    def greedySearch(self, images: Tensor):
        """_summary_

        Args:
            images (Tensor): _description_

        Returns:
            _type_: _description_
        """
        sequences = torch.tensor([[self.word2id['startseq']]] *
                                 images.shape[0]).to(images.device)
        for i in range(self.max_length - 1):
            seqs_pad = torch.nn.functional.pad(sequences,
                                               (self.max_length - i - 1, 0),
                                               value=0)

            seqs_pad = seqs_pad.to(images.device)
            pred = self(images, seqs_pad)
            pred = torch.argmax(pred, dim=1, keepdim=True)
            sequences = torch.cat((sequences, pred), dim=1)

        captions = []
        for sequence in sequences:
            caption = []
            for id in sequence:
                w = self.id2word[id.cpu().item()]
                if w == 'endseq': break
                caption.append(w)
            captions.append(' '.join(caption[1:]))

        return captions


if __name__ == "__main__":
    net = ImageCaptionNet(image_embed_net=InceptionNet(device='cpu'),
                          text_embed_net=Glove_RNN())

    sequences = torch.randint(0, 100, (20, 2))
    images = torch.randn(2, 3, 299, 299)
    out = net(images, sequences)
    print(out.shape)