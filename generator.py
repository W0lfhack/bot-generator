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

    dataloader_testset = DataLoader(mnist_testset, batch_size=64, shuffle=False)
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

def train_step_discriminator(epoch_index,     tb_writer, 
                             training_loader,           generator,
                             discriminator, g_optimizer, 
                             d_optimizer):
    running_loss_d = 0
    last_loss_d = 0
    running_loss_g = 0
    last_loss_g = 0

    for i, data in enumerate(training_loader):
        inputs, _ = data
        inputs = inputs.view(inputs.size(0), -1) 

        # Discriminator training
        d_optimizer.zero_grad()

        outputs_d = discriminator(inputs)

        targets_ones = torch.ones_like(outputs_d).to(outputs_d.device)
        noise = torch.randn(inputs.size(0), 100).to(inputs.device)

        fake_images = generator(noise)

        outputs_fake = discriminator(fake_images.detach())
        targets_zeros = torch.zeros_like(outputs_fake).to(outputs_fake.device)

        criterion = nn.BCEWithLogitsLoss()

        loss_ones = criterion(outputs_d, targets_ones)
        loss_zeros = criterion(outputs_fake, targets_zeros)
        loss_d = loss_ones + loss_zeros

        loss_d.backward()
        d_optimizer.step()

        running_loss_d += loss_d.item()
        if i % 1000 == 999:
            last_loss_d = running_loss_d / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss_d}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss_d, tb_x)
            running_loss_d = 0.

        # Generator training
        g_optimizer.zero_grad()

        outputs_fake_for_g = discriminator(fake_images)
        targets_ones_g = torch.ones_like(outputs_fake_for_g).to(outputs_fake_for_g.device)
        
        loss_g = criterion(outputs_fake_for_g, targets_ones_g)

        loss_g.backward()
        g_optimizer.step()

        running_loss_g += loss_g.item()
        if i % 1000 == 999:
            last_loss_g = running_loss_g / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss_g}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss_g, tb_x)
            running_loss_g = 0.

    return (last_loss_g, last_loss_d)

if __name__ == "__main__":
    main()