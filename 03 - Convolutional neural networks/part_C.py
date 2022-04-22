from abc import ABC
import torch
import torch.nn as nn
from Utils.mnist import getTestData, getTrainingData

# Traning values
LEARNING_RATE = 0.001
EPOCHS = 5


# Creating model
class ConvolutionalNeuralNetworkModel(nn.Module, ABC):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        self.logits = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Dropout(p=0.2),
                                    nn.Conv2d(32, 64, kernel_size=5, padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(64 * 7 * 7, 1024),
                                    nn.Linear(1024 * 1 * 1, 10))

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()

# Get training and Test data
x_train, y_train = getTrainingData()
x_test, y_test = getTestData()

# Normalize input
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)

# Optimizing model
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
print("|*Epoch*|*Accuracy*|\n|:---:|:---:|")
for epoch in range(EPOCHS):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
        optimizer.step()
        optimizer.zero_grad()

    # Check accuracy
    accuracy = model.accuracy(x_test, y_test).item()
    print("|%s|%.2f%%|" % (epoch + 1, accuracy * 100))
