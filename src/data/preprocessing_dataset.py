from typing import List, Optional

import torch
import imageio
import numpy as np
import os.path as osp
from pickle import dump, load
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence

class PreprocessingDataset(Dataset):
    # glove: wget https://nlp.stanford.edu/data/glove.6B.zip
    def __init__(self, 
                 dataset: Dataset,
                 word_count_threshold: int = 10,
                 vocab: List[str] = None,
                 max_length: int = None,
                 save_weight_embedding: bool = False,
                 transform: Optional[T.Compose] = None):
        
        if vocab is None:
            word_counts = {}  # a dict : { word : number of appearances}
            max_length = 0
            for i in range(len(dataset)):
                _, captions = dataset[i]
                for caption in captions:
                    words = caption.split()
                    max_length = len(words) if (max_length < len(words)) else max_length
                    for w in words:
                        try:
                            word_counts[w] += 1
                        except:
                            word_counts[w] = 1
                            
            self.vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        else: self.vocab = vocab

        self.i2w, self.w2i, id = {}, {}, 1
        for w in self.vocab:
            self.w2i[w] = id
            self.i2w[id] = w
            id += 1

        if save_weight_embedding:
            self.save_weight_embedding(glove_dir = 'glove')

        self.dataset = dataset
        self.max_length = max_length

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize([299, 299], antialias=True), # using inception_v3 to encode image
            ])

    def save_weight_embedding(self, glove_dir, embedding_dim = 200):
        embedding_matrix_path = osp.join(glove_dir, 'embedding_matrix.pkl')
        if osp.exists(embedding_matrix_path):
            return
        
        file = open(osp.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")
        embeddings_index = {}
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            coefs = torch.from_numpy(coefs)
            embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(embeddings_index))

        print(embeddings_index['unknown'])
        print(embeddings_index[' '])

        embedding_matrix = torch.zeros((self.vocab_size(), embedding_dim))
        for word, i in self.w2i.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector
        print('Embedding matrix:', embedding_matrix.shape)

        # Open a file for writing with binary mode
        with open(embedding_matrix_path, "wb") as file:
            dump(embedding_matrix, file)
    
    def __len__(self):
        return len(self.dataset)
    
    def vocab_size(self):
        return len(self.vocab) + 1 # add padding
    
    def __getitem__(self, idx):
        img_path, captions = self.dataset[idx]

        sequences = []
        for caption in captions:
            seq = [self.w2i[word] for word in caption.split(' ') if word in self.w2i]
            seq = pad_sequence([torch.tensor(seq), torch.zeros(self.max_length)])
            sequences.append(seq[:, 0])
        sequences = torch.stack(sequences, dim=0)

        image = imageio.v2.imread(img_path)
        image = self.transform(image)
        return image, sequences

if __name__ == "__main__":
    import pyrootutils
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from src.data.dataset import FlickrDataset8k

    dataset = FlickrDataset8k()
    preprocessing_dataset = PreprocessingDataset(dataset, 
                                                 save_weight_embedding=True)
    image, captions = preprocessing_dataset[0]
    print(image.shape)
    print(captions.shape)

    # Open a file for reading with binary mode
    with open("glove/embedding_matrix.pkl", "rb") as file:
        embedding_matrix = load(file)
        print('Embedding_matrix:', embedding_matrix.shape)
