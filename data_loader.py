import os
import os.path as osp
import glob

from torch.utils import data
from torchvision import transforms as T
from PIL import Image

class DataSet(data.Dataset):

    def __init__(self, config, img_transform):
        self.img_transform = img_transform
        self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], config['TRAINING_CONFIG']['MODE'])

        self.data_list = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.data_list = [x.split(os.sep)[-1].split('_')[0] for x in self.data_list]
        self.data_list = list(set(self.data_list))

    def __getitem__(self, index):
        fid = self.data_list[index]
        color = Image.open(osp.join(self.img_dir, '{}_color.png'.format(fid)))
        sketch = Image.open(osp.join(self.img_dir, '{}_sketch.png'.format(fid)))
        return fid, self.img_transform(color), self.img_transform(sketch)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(config):

    img_transform = list()
    img_size = config['MODEL_CONFIG']['IMG_SIZE']

    img_transform.append(T.Resize((img_size, img_size)))
    img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    dataset = DataSet(config, img_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
