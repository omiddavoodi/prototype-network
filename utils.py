# Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions
# Li et al., 2018 (https://ojs.aaai.org/index.php/AAAI/article/view/11771)
#
# MIT License
#
# Copyright (c) [2024] [Omid Davoodi]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def data_loaders(batch_size, data_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=transform)
    train_size = int(0.9 * len(trainset))
    test_size = len(trainset) - train_size
    trainset, validset = torch.utils.data.random_split(trainset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=6)
    testset = torchvision.datasets.MNIST(data_path, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, validloader, testloader

def imshow(img):
    img = img / 2 + 0.5     # Scale the values between 0 and 1 from between -1 and 1
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def prototype_loss(weight_vector, outputs, labels, decoded, inputs, c):
    cl = nn.CrossEntropyLoss()(outputs, labels)
    ml = nn.MSELoss()(decoded, inputs)
    r1 = torch.mean(torch.min(c, 0)[0])
    r2 = torch.mean(torch.min(c, 1)[0])
    loss = weight_vector[0] * cl + weight_vector[1] * ml + weight_vector[2] * r1 + weight_vector[3] * r2
    return loss