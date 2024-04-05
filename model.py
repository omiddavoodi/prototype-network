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
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, stddev=0.1):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 10, 3, stride=2)
        self.relu4 = nn.ReLU()
        self._init_weights(stddev)

    def forward(self, x):
        x = F.pad(x, (0, 2, 0, 2), mode='replicate')
        x = self.relu1(self.conv1(x))
        x = F.pad(x, (0, 2, 0, 2), mode='replicate')
        x = self.relu2(self.conv2(x))
        x = F.pad(x, (0, 2, 0, 2), mode='replicate')
        x = self.relu3(self.conv3(x))
        x = F.pad(x, (0, 2, 0, 2), mode='replicate')
        x = self.relu4(self.conv4(x))
        return x

    def _init_weights(self, stddev):
        nn.init.normal_(self.conv1.weight, std=stddev)
        nn.init.normal_(self.conv2.weight, std=stddev)
        nn.init.normal_(self.conv3.weight, std=stddev)
        nn.init.normal_(self.conv4.weight, std=stddev)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(10, 32, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        return x

# The overall network
class Net(nn.Module):
    def __init__(self, num_prototypes=30, num_classes=2, stddev=0.1):
        super(Net, self).__init__()
        self.encoder = Encoder(stddev)
        self.decoder = Decoder()
        self.prototypes = nn.Parameter(torch.normal(torch.zeros(num_prototypes, 10*2*2), stddev))
        self.fc = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        # y is autoencoder's reconstruction

        x = x.view(-1, 10*2*2)
        k2 = x.detach()

        # Calculate distances between prototypes and the encodings
        # Distance is cauculated by calculateing x**2 + y**2 - 2xy instead of (x-y)**2
        xx = torch.sum(x.pow(2), 1)
        pp = torch.sum(self.prototypes.pow(2), 1)
        c = torch.neg(torch.matmul(x, self.prototypes.T)) * 2
        c = c + xx[:,None]
        c = c + pp[None,:]

        # feed that distance to the last layer
        x = torch.sigmoid(self.fc(c))

        return x, y, c, k2