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
import torch.optim as optim
import torchvision
import argparse
from model import Net
from utils import data_loaders, imshow, prototype_loss

def train(net, device, trainloader, optimizer, weight_vector, num_epochs=40, save_path='./', save_name='trained_model.pth'):
    net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, decoded, c, _ = net(inputs)
            loss = prototype_loss(weight_vector, outputs, labels, decoded, inputs, c)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i > 0 and i % 100 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    prots = net.prototypes
    closest_prots = torch.zeros_like(prots).to(device)
    closest_inputs = torch.zeros(prots.shape[0], 1, 28, 28).to(device)
    closest_labels = torch.zeros(prots.shape[0], dtype=torch.long).to(device)
    first = True

    with torch.no_grad():
        # Go through the training data and find the closest actual data sample to each prototype and assign the prototype to that point in the embedding space
        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            _, _, c, x = net(inputs)

            c2 = torch.sum((prots - closest_prots).pow(2), 1)
            indices = torch.argmin(c.T, axis=1)
            mindists = c.T.gather(1, indices.view(-1, 1)).view(-1)

            if first:
                changes = c2 == c2
                first = False
            else:
                changes = c2 > mindists

            closest_prots[changes] = x[indices[changes]]
            closest_inputs[changes] = inputs[indices[changes]]
            closest_labels[changes] = labels[indices[changes]]

        prots_vis = closest_inputs
        net.prototypes.weight = closest_prots
    
        # Show the prototype images in a grid
        imshow(torchvision.utils.make_grid(prots_vis.cpu()))
    print('Finished Prototype Assignment')

    # Save the trained model
    torch.save(net.state_dict(), f"{save_path}/{save_name}")

def evaluate(net, device, testloader):
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs, _, _, _ = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %.2f%%' % (100 * correct / total))

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate prototype network')
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 250)')
    parser.add_argument('--data-path', type=str, default='./data', metavar='PATH',
                        help='path to dataset (default: ./data)')
    parser.add_argument('--num-epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='learning rate (default: 0.002)')
    parser.add_argument('--num-prototypes', type=int, default=30, metavar='N',
                        help='number of prototypes in the model (default: 30)')
    parser.add_argument('--save-path', type=str, default='./', metavar='PATH',
                        help='path to save the trained model (default: ./)')
    parser.add_argument('--save-name', type=str, default='trained_model.pth', metavar='NAME',
                        help='name of the saved model file (default: trained_model.pth)')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, validloader, testloader = data_loaders(args.batch_size, args.data_path)

    num_prototypes = args.num_prototypes
    num_classes = 10 # For our specific MNIST implementation
    stddev = 0.1 # For initializing the network weights.

    # The weight vector encodes the weights of the subcomponents of the loss function.
    weight_vector_size = 4
    weight_vector = torch.ones(weight_vector_size, device=device)
    weight_vector[0] = 1    # Cross-entropy loss
    weight_vector[1] = 0.05 # auto-encoder reconstruction loss. Lambda in the paper
    weight_vector[2] = 0.05 # r1 loss. Lambda-1 in the paper
    weight_vector[3] = 0.05 # r2 loss. Lambda-2 in the paper

    net = Net(num_prototypes, num_classes, stddev).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    train(net, device, trainloader, optimizer, weight_vector, args.num_epochs, args.save_path, args.save_name)

    # Accuracy results
    evaluate(net, device, testloader)

if __name__ == '__main__':
    main()