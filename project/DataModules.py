import os, json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class CityScapesDataset(Dataset):
    def __init__(self, root_dir, transform):

        self.real_dir = root_dir + "images_real/"
        self.noise_dir = root_dir + "images_noise/"
        self.bbox_dir = root_dir + "bbox/"
        self.transform = transform

        self.real = sorted(os.listdir(self.real_dir))
        self.noise = sorted(os.listdir(self.noise_dir))
        self.bboxes = sorted(os.listdir(self.bbox_dir))

    def __getitem__(self, index):
        real = Image.open(self.real_dir + self.real[index])
        real = self.transform(real)
        noise = Image.open(self.noise_dir + self.noise[index])
        noise = self.transform(noise)
        f = open(self.bbox_dir + self.bboxes[index])
        bbox_dict = json.load(f) # za učitati info iz bbox datoteke
        f.close()

        return {'real': real, 'noise': noise, 'bbox': bbox_dict}
    
    def __len__(self):
        return len(self.real)

class Cityscapes(pl.LightningDataModule):
    def __init__(self, data_dir='./data/', batch_size=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # ako je bez normalizacije
        # self.transform = transforms.ToTensor()
        self.transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = CityScapesDataset(self.data_dir + 'train/', self.transform)
        if stage == 'test' or stage is None:
            self.test = CityScapesDataset(self.data_dir + 'test/', self.transform)
        # u kodu nema validation
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
