import matplotlib.pyplot as plt
from Utils.mnist import getTrainingNumbers, getTestNumbers
import torch
import imageio
import os

epochs_ = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
for i in epochs_:
    # Initial Values
    W_init = (784, 10)  # Number of images, number of classes
    b_init = (1, 10)

    # Training values
    learning_rate = 0.1
    epochs = i

    # Creating model
    class NumberModel:
        def __init__(self):
            self.m = torch.nn.Softmax(dim=1)
            self.W = torch.ones(W_init, requires_grad=True)
            self.b = torch.ones(b_init, requires_grad=True)

        def f(self, x):
            return self.m(x @ self.W + self.b)

        def loss(self, x, y):
            return torch.nn.functional.cross_entropy(self.f(x), y.argmax(1))

        def accuracy(self, x, y):
            return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


    model = NumberModel()

    # Trainging data
    x_train, y_train = getTrainingNumbers()

    optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    # Testdata
    x_test, y_test = getTestNumbers()

    # Check accuracy
    loss = model.loss(x_train, y_train).detach().numpy()
    accuracy = model.accuracy(x_test, y_test).detach().numpy()

    # Visualize W
    fig = plt.figure(figsize=(8, 8))
    for j in range(1, 11):
        img = model.W.detach().numpy()[:, j - 1].reshape(28, 28)
        fig.add_subplot(3, 4, j)
        plt.imshow(img)

    # Show values and save image
    fig.text(0.55, 0.15, "epochs    = %.0f\nAccuracy = %.1f%%" % (epochs, accuracy * 100)).set_fontsize(25)
    plt.savefig("images/gif/" + str(epochs) + "_epochs.png")
    plt.show()

# Creating gif of images
images = []
for filename in os.listdir("./images/gif"):
    images.append(imageio.imread("images/gif/" + filename))
imageio.mimsave('./images/test.gif', images, duration=0.8, loop=5)
