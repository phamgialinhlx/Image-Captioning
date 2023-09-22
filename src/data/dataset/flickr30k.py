import glob
import imageio
import pandas as pd
import os.path as osp
from torch.utils.data import Dataset


class FlickrDataset30k(Dataset):

    dataset_dir = 'flickr30k'
    dataset_url = 'https://www.kaggle.com/datasets/eeshawn/flickr30k'

    def __init__(self,
                 data_dir: str = 'data') -> None:
        """
            data_dir:
        """
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.df = pd.read_csv(osp.join(self.dataset_dir, 'captions.txt'))

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.df['caption'][index]
        img_path = osp.join(self.dataset_dir, 'Images', self.df['image'][index])
        
        image = imageio.v2.imread(img_path)
        
        return image, caption
    
if __name__ == "__main__":
    dataset = FlickrDataset30k()
    print(len(dataset))
    image, caption = dataset[0]
    print('Image size:', image.shape)
    print('Caption:', caption)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()