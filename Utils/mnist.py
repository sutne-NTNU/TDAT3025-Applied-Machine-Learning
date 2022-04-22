import torch
import torchvision


def getTrainingNumbers():  # 02
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 784).float()
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1
    return [x_train, y_train]


def getTestNumbers():  # 02
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 784).float()
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1
    return [x_test, y_test]


def getTrainingData():  # 03
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1
    return [x_train, y_train]


def getTestData():  # 03
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1
    return [x_test, y_test]


def getTrainingFashion():  # 03
    mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1
    return [x_train, y_train]


def getTestFashion():  # 03
    mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1
    return [x_test, y_test]
