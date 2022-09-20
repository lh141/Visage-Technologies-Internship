import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128, num_workers=int(os.cpu_count() / 2)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) # očekivanje i std. devijacija za MNIST
        ])

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(50, 1) # samo real/fake -> 1 izlaz
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = x.view(-1, 320) # spljošti ga da moze u fully connected slojeve

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # 7x7 slike sa 64 feature mapa
        self.lin1 = nn.Linear(latent_dim, 7*7*64)
        # Conv2D smanjuje tenzor, ConvTranspose2D povećava tenzor
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.conv = nn.Conv2d(16, 1, kernel_size=7)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = self.relu(x)
        x = x.view(-1, 64, 7, 7)
        
        # Transposed convolution to 16x16 (64 feature maps)
        # izračun dimenzije izlaza: http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html
        x = self.ct1(x)
        x = self.relu(x)
        
        # Transposed convolution to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = self.relu(x)
        
        # Convolution to 28x28 (1 feature map)
        return self.conv(x)


class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002, batch_size=64):
        super().__init__()
        self.save_hyperparameters() # sprema parametre iz poziva u model.hparams

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        # za provjeru generatora, kak napreduje u stvaranju slika iz skupa latentnih točaka
        self.validation_z = torch.randn(10, self.hparams.latent_dim)

        self.loss = nn.BCELoss()

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return self.loss(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch

        # sample noise
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)

        # train generator
        if optimizer_idx == 0: # configure_optimizers vraca [opt_g, opt_d]
            self.generated_imgs = self(z) # jer forward poziva generator
            predictions = self.discriminator(self.generated_imgs)
            # stavimo da su generirane slike prave da bude veći gubitak ako je diskriminator točno odredio da su generirane
            g_loss = self.adversarial_loss(predictions, torch.ones(real_imgs.size(0), 1))

            # log sampled images TENSORBOARD
            sample_imgs = self.generated_imgs[:10]
            grid = torchvision.utils.make_grid(sample_imgs, nrow=5)
            tensorboard = self.logger.experiment
            tensorboard.add_image("generated_images", grid, 0)

            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            real_preds = self.discriminator(real_imgs)
            real_loss = self.adversarial_loss(real_preds, torch.ones(real_imgs.size(0), 1))

            fake_preds = self.discriminator(self(z).detach())
            fake_loss = self.adversarial_loss(fake_preds, torch.zeros(real_imgs.size(0), 1)) 


            d_loss = (real_loss + fake_loss) / 2

            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss
    
    def on_epoch_end(self):
        # log sampled images
        sample_imgs = self(self.validation_z)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=5)
        tensorboard = self.logger.experiment
        tensorboard.add_image("generated_images", grid, self.current_epoch)

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []


trainer = pl.Trainer(max_epochs=30)
dm = MNISTDataModule()
model = GAN()
trainer.fit(model, dm)