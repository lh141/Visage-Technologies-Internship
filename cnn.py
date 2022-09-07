import enum
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"

class ConvNet(nn.Module):
    def __init__(self, numChannels, classes):
    # crno-bijelo NC=1, u boji NC=3 (red, green, blue)
    # classes - broj finalnih labela
        super(ConvNet, self).__init__()

        # može se definirati i pomoću Sequentiala, sad preko klasa 

        # prvi set CONV => RELU => POOL slojeva
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=32, kernel_size=(3, 3), padding=1)
        # 20 različitih filtera, svaki dimenzija 3x3
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # 2x2 je dimenzija filtera za maxpooling, stride je 2 u svakom smjeru

        # drugi set CONV => RELU => POOL slojeva
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # 2x2 je dimenzija filtera za maxpooling, stride je 2 u svakom smjeru

        self.flatten = nn.Flatten()

        # set FC => RELU slojeva
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.relu3 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=600, out_features=120)
        self.relu4 = nn.ReLU()

        # zadnji set - softmax + klasifikacija
        self.fc2 = nn.Linear(in_features=120, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        # konačni zbroj mora biti 1 u dimenziji 1:
        # dimenzija 0 - broj batcheva, dimenzija 1: broj uzoraka u batchu

    
    def forward(self, x):
        # prvi set CONV => RELU => POOL slojeva
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # drugi set CONV => RELU => POOL slojeva
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # setovi FC => RELU slojeva, ali prvo ga spljošti
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc3(x)
        x = self.relu4(x)

        # idemo u klasifikaciju pa u softmax
        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output

# Hyperparameters: brzina ucenja, velicina uzorka, broj slojeva
learning_rate = 1e-3
batch_size = 64
epochs = 5

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = { # DICTIONARY
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# računanje broja training vs. validation podataka
train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


model = ConvNet(numChannels=1, classes=len(labels_map)).to(device)

# loss funkcija, u kombinaciji s logsoftmaxom u definiciji modela daje CrossEntropyLoss
loss_fn = nn.NLLLoss()
# funkcija za optimizaciju parametara
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    totalTrainLoss = 0
    trainCorrect = 0
    for (X, y) in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)

        # najbitnija tri koraka u točno tom redoslijedu!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        # argmax(1) vrati maksimum vrijednosti s indeksom 1
        # tj. ono kaj mreža misli da je slika
    
    trainCorrect /= len(dataloader.dataset)
    totalTrainLoss /= len(dataloader) # ukupni gubitak/broj batcheva
    print(f"Train..... Accuracy: {(100*trainCorrect):>0.1f}%, Avg loss: {totalTrainLoss:>8f}")


def test_loop(epoch, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        model.eval()

        for (X, y) in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test..... Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", 100*correct, epoch)
    writer.flush()

    return 100*correct

epoch_acc = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    epoch_acc += test_loop(t, test_dataloader, model, loss_fn)

epoch_acc = epoch_acc / epochs
writer.add_hparams({'lr': learning_rate, 'batch_s': batch_size, 'epochs': epochs, 'loss': 'NLL', 'optim': 'Adam'},
                    {'accuracy': epoch_acc})
writer.add_graph(model, iter(train_dataloader).next()[0])    

writer.close()
print("Done!")