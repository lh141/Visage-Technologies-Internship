import enum
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


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

labels_map = { #DICTIONARY
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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self): #__tekst__ -> mozes pozivati sa ()
        super(NeuralNetwork, self).__init__() #mozes pristupiti onome kaj naslijedi od roditeljske klase
        self.flatten = nn.Flatten() # spljosti 2D u 1D
        self.linear_relu_stack = nn.Sequential( #lista slojeva kroz koje se prolazi
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits #vrijednosti koje vraca zadnji sloj prije nego ode u softmax


model = NeuralNetwork()

#Hyperparameters: brzina ucenja, velicina uzorka, broj slojeva
learning_rate = 1e+2
batch_size = 64
epochs = 10

#loss funkcija
loss_fn = nn.CrossEntropyLoss()
#funkcija za optimizaciju parametara
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(epoch, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader): #X uvijek input variables, y output class labels
        pred = model(X) #prediction
        loss = loss_fn(pred, y) #gubitak s obzirom na prediction i ono kaj stvarno je

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
def test_loop(epoch, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss/= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", 100*correct, epoch)
    writer.flush()

    return 100*correct

epoch_acc = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(t, train_dataloader, model, loss_fn, optimizer)
    epoch_acc += test_loop(t, test_dataloader, model, loss_fn)
epoch_acc = epoch_acc / epochs
writer.add_hparams({'lr': learning_rate, 'batch_s': batch_size, 'epochs': epochs, 'loss': 'CS', 'optim': 'SGD'},
                    {'accuracy': epoch_acc})
writer.add_graph(model, iter(train_dataloader).next()[0])    
print("Done!")
writer.close()