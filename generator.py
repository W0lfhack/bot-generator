import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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
        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=1),
                                    nn.Sigmoid(),
                                    nn.BCELoss())
                                    

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

        
def optimizer(G, D):
    optimizerG = torch.optim.Adam(G.parameters())
    optimizerD = torch.optim.Adam(D.parameters())

    return (optimizerG, optimizerD)

def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5],[0.5])])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    G = Generator()
    D = Discriminator()

    print(optimizer(G,D))
    

if __name__ == "__main__":
    main()