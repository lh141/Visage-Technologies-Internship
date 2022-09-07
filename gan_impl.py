import torch
import argparse
import os
import numpy as np
import math

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F

from typing import List
from matplotlib.colors import ListedColormap as colormap

torch.manual_seed(111)

parser = argparse.ArgumentParser() 
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
#parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr_D", type=float, default=0.03, help="SGD: learning rate (generator)")
parser.add_argument("--lr_G", type=float, default=0.1, help="SGD: learning rate (discriminator)")
#parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
#parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
#parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first order momentum of gradient")
#parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of first order momentum of gradient")
#parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
#parser.add_argument("--channels", type=int, default=1, help="number of image channels")
#parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

#priprema pravih podataka

def randNormal():
    u = 1 - np.random.uniform(0, 1)
    v = 1 - np.random.uniform(0, 1)
    return math.sqrt(-2.0*math.log(u)) * math.cos(2.0*math.pi*v)

def generate_ring_data(atlas_size: int) -> List[List[float]]:
    data = []
    for i in range(0, atlas_size):
        rand = np.random.uniform(0, 1)
        data.append([
            0.5 + 0.3 * math.cos(rand * math.pi * 2) +
            0.025 * randNormal(),
            0.45 + 0.25 * math.sin(rand * math.pi * 2) +
            0.025 * randNormal(),
        ])
    return np.array(data)

def generate_random_2d_uniform(atlas_size: int, noise_size=2):
    return np.random.uniform(0, 1, [atlas_size, noise_size])

train_data_length = 600
train_labels = torch.zeros(train_data_length)

ring_dataset = torch.from_numpy(generate_ring_data(train_data_length))
plt.scatter(ring_dataset[:, 0], ring_dataset[:,1], color="green")
plt.show()

train_set = [
    (ring_dataset[i], train_labels[i]) for i in range(train_data_length)
]
batch_size = int(train_data_length/3)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True 
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        #m.weight.data.normal_(0, 1.0/math.sqrt(2))
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

#Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential( #1 hidden layer, 10 neurons
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Sigmoid(),
        )
        self.model.apply(init_weights)
    
    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential( #2 hidden layers, 9 neurons
            nn.Linear(2, 9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, 1), #1 jer je real ili fake
            nn.Sigmoid(),
        )
        self.model.apply(init_weights)

    def forward(self, x):
        output = self.model(x.float())
        return output

discriminator = Discriminator()


loss_fn = nn.BCELoss() #Log loss
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=opt.lr_D)
optimizer_G = torch.optim.SGD(generator.parameters(), lr=opt.lr_G)

def arrange_and_cartesian_prod(first=0, last=1, step=0.01):
    x = np.arange(first, last, step)
    x1, x2 = np.meshgrid(x, x)
    return np.stack([x1.ravel(), x2.ravel()]).T

grid = torch.from_numpy(arrange_and_cartesian_prod())

#treniranje mreza

for epoch in range(opt.n_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        #podaci za diskriminator - i pravi i lazni
        real_samples_labels = torch.ones((batch_size, 1)) #sve su jedinice za prave uzorke
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(latent_space_samples) #provedemo latent_space_samples kroz generator
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        #treniranje diskriminatora
        discriminator.zero_grad() #u svakom koraku trebamo postaviti gradijente na nulu da se ne zbroje
        output_D = discriminator(all_samples)
        loss_D = loss_fn(
            output_D, all_samples_labels
        )
        loss_D.backward()
        optimizer_D.step()

        #podaci za generator
        latent_space_samples = torch.randn((batch_size, 2))

        #treniranje generatora
        generator.zero_grad() #u svakom koraku trebamo postaviti gradijente na nulu da se ne zbroje
        generated_samples = generator(latent_space_samples)
        output_D_generated = discriminator(generated_samples)
        loss_G = loss_fn(
            output_D_generated, real_samples_labels #zelimo da bude cim uvjerljivije, veci loss je ako je dalje od 1
        )
        loss_G.backward()
        optimizer_G.step()

    #ispisi gubitak
    if epoch % 100 == 0 or epoch == opt.n_epochs - 1:
        color_map = []
        with torch.no_grad():
            grid_D = discriminator(grid)
            #torch.sigmoid_(grid_D)
        grid_D1 = torch.ones_like(grid_D)
        grid_D1.copy_(grid_D)
        for i in range(10000):
            if grid_D[i] < 0.5:
                grid_D1[i] = 1 - 2 * grid_D[i]
            elif grid_D[i] > 0.5:
                grid_D1[i] = 2 * grid_D[i] - 1
            else:
                grid_D1[i] = 0

        generated_samples = generated_samples.detach()
        plt.scatter(grid[:, 0], grid[:, 1], c=grid_D, cmap=colormap(['green', 'purple']), alpha=grid_D1.numpy())
        plt.scatter(ring_dataset[:, 0], ring_dataset[:,1], color="green")
        plt.scatter(generated_samples[:, 0], generated_samples[:,1], color="purple")
        plt.show()
        print(f"Epoch: {epoch} Loss D.: {loss_D}")
        print(f"Epoch: {epoch} Loss G.: {loss_G}")