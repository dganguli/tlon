import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layers = OrderedDict([
            ('conv_1', nn.Conv2d(1, 10, kernel_size=5)),
            ('max_pool2d_1', nn.MaxPool2d(2)),
            ('relu_1', nn.ReLU()),
            ('conv_2', nn.Conv2d(10, 20, kernel_size=5)),
            ('dropout_2d', nn.Dropout2d()),
            ('max_pool2d_2', nn.MaxPool2d(2)),
            ('relu_2', nn.ReLU()),
            ('reshape', View((-1, 320))),
            ('fc1', nn.Linear(320, 50)),
            ('dropout_1d', nn.Dropout()),
            ('fc2', nn.Linear(50, 10)),
            ('log_softmax', nn.LogSoftmax(dim=1))
        ])
        self.model = nn.Sequential(self.layers)

    def forward(self, x):
        return self.model(x)

    def get_activation_layer(self, x, layer_name):
        self.eval()
        sub_model_layers = []

        if layer_name not in self.layers:
            raise ValueError('layer {} not found in model'.format(layer_name))
        else:
            for current_layer_name, layer in self.layers.items():
                sub_model_layers.append(layer)
                if current_layer_name == layer_name:
                    break

        model = nn.Sequential(sub_model_layers)

        with torch.no_grad():
            out = model(x)

        return out

    def from_save_dict(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()
        return self


class CNNTrainer:
    def __init__(self,
                 train_loader,
                 test_loader,
                 save_path=None,
                 n_epochs=3,
                 learning_rate=0.01,
                 momentum=0.5):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_path = save_path

        self.n_epochs = n_epochs

        self.CNN = CNN()

        self.optimizer = optim.SGD(self.CNN.parameters(),
                                   lr=learning_rate,
                                   momentum=momentum)

        self.cuda = True if torch.cuda.is_available() else False
        if self.cuda:
            self.CNN.cuda()

        self.train_losses = []
        self.train_counter = []

        self.test_losses = []
        self.test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def update(self, epoch, log_interval):

        self.CNN.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if self.cuda:
                data = data.cuda()
                target = target.cuda()
            output = self.CNN(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(self.train_loader.dataset)))
                if self.save_path is not None:
                    torch.save(self.CNN.state_dict(),
                               os.path.join(self.save_path, 'model.pth')
                               )
                    torch.save(self.optimizer.state_dict(),
                               os.path.join(self.save_path, 'optimizer.pth')
                               )

    def test(self):
        self.CNN.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                if self.cuda:
                    data = data.cuda()
                    target = target.cuda()
                output = self.CNN(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def train(self, log_interval=10):
        self.test()
        for epoch in range(1, self.n_epochs + 1):
            self.update(epoch, log_interval)
            self.test()


if __name__ == '__main__':
    from data import load_mnist
    import time

    train_loader, test_loader = load_mnist('/tmp')
    trainer = CNNTrainer(train_loader, test_loader, save_path='/tmp')
    start = time.time()
    trainer.train(100)
    end = time.time()
    elapsed = end - start
    print(elapsed)
