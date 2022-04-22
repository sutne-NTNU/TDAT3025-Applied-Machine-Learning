from Utils.mnist import getTestFashion, getTrainingFashion
import matplotlib.pyplot as plt
import torch.nn as nn
from abc import ABC
import imageio
import torch
import os

# Traning values
LEARNING_RATE = 0.001
EPOCHS = 5


# Creating model
class ConvolutionalNeuralNetworkModel(nn.Module, ABC):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        self.logits = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Conv2d(32, 64, kernel_size=5, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Flatten(),
                                    nn.Linear(64 * 7 * 7, 10))

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
x_train, y_train = getTrainingFashion()
x_test, y_test = getTestFashion()

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

#     # Visualize Model
#     fig = plt.figure(figsize=(8, 8))
#     for i in range(1, 11):
#         img = param.data.detach().numpy()[:, i - 1].reshape(28, 28)
#         fig.add_subplot(3, 4, i)
#         plt.imshow(img)
#
#     # Add values
#     fig.text(0.55, 0.15, "epoch    = %.0f\nAccuracy = %.1f%%" % (epoch + 1, accuracy * 100)).set_fontsize(25)
#     # Save image
#     plt.savefig("images/gif/epoch_" + str(epoch + 1) + ".png")
#     plt.show()
#
# # Creating gif of images
# images = []
# for filename in os.listdir("./images/gif"):
#     images.append(imageio.imread("images/gif/" + filename))
# imageio.mimsave('./test.gif', images, duration=1.5, loop=5)
