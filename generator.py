import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(in_features=100, out_features=256),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(in_features=256, out_features=512),
                                            nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                            nn.LeakyReLU())
        self.output = nn.Sequential(nn.Linear(in_features=1024, out_features=28*28),
                                    nn.Tanh())
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(in_features=784, out_features=1024),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                    nn.LeakyReLU())
        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=1))
                                    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

def dataloader():
    transform = v2.Compose([transforms.ToImage(),
                            transforms.ToDtype(torch.float32, scale=True),
                            transforms.Normalize([0.5,],[0.5,])])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    dataloader_testset = DataLoader(mnist_testset, batch_size=64, shuffle=True)
    dataloader_trainset = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    return (dataloader_testset, dataloader_trainset)
def main():
    dataloader_testset = dataloader()[0]
    dataloader_trainset = dataloader()[1]

    G = Generator()
    D = Discriminator()
    loss = nn.BCEWithLogitsLoss()
    optimizerG = torch.optim.Adam(G.parameters())
    optimizerG = torch.optim.Adam(D.parameters())

    

if __name__ == "__main__":
    main()