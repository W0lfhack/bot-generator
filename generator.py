import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        return img

def main():
    transform = transforms.ToTensor()
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print(mnist_trainset)
    print(len(mnist_testset))
    generator = Generator(100, 28)

if __name__ == "__main__":
    main()