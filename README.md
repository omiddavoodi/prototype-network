# Prototype Network

Pytorch implementation for "[Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions](https://ojs.aaai.org/index.php/AAAI/article/view/11771)" by Li et al.

# Requirements

* torch
* torchvision
* numpy
* matplotlib
* pillow

You can run this command to install all of the requirements:

```bash
$ python -m pip install -r requirements.txt
```

# Usage

```bash
usage: train.py [-h] [--batch-size N] [--data-path PATH] [--num-epochs N] [--lr LR]
                [--num-prototypes N] [--save-path PATH] [--save-name NAME]

Train and evaluate prototype network

optional arguments:
  -h, --help          show this help message and exit
  --batch-size N      input batch size for training (default: 250)
  --data-path PATH    path to dataset (default: ./data)
  --num-epochs N      number of epochs to train (default: 40)
  --lr LR             learning rate (default: 0.002)
  --num-prototypes N  number of prototypes in the model (default: 30)
  --save-path PATH    path to save the trained model (default: ./)
  --save-name NAME    name of the saved model file (default: trained_model.pth)
```

Use the following command to train a model on MNIST.

```bash
python train.py
```

# License

This repository is released under the MIT license.