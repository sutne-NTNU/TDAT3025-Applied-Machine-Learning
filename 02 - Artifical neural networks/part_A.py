import matplotlib.pyplot as plt
import torch

# Training data
x_train = torch.tensor([[0], [1]])  # input
y_train = torch.tensor([[1], [0]])  # expected output

# Initial values of W and b
W_init = [0.0]
b_init = [0.0]

# Training values
learning_rate = 5
epochs = 1_000_000


# Creating model
class SigmoidNOTModel:
    def __init__(self):
        self.W = torch.tensor([W_init], requires_grad=True)
        self.b = torch.tensor([b_init], requires_grad=True)

    # Predicotor
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    # Uses Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = SigmoidNOTModel()

# Optimize the model
optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Plot training data
plt.plot(x_train, y_train, 'o', c="blue", label="$(\\hat x^{(i)},\\hat y^{(i)})$")

# Plot result
line_xvalues = torch.linspace(0, 1).reshape(-1, 1)
plt.plot(line_xvalues, model.f(line_xvalues).detach(), color="green", label="$f(x) = sigmoid(xW+b)$")
plt.legend()
plt.xlabel("Learning rate: %s       Number of Epochs: %s\nW = %.3f\nb = %.3f\nloss = %.3f"
           % (learning_rate, epochs,
              round(model.W.item(), 3), round(model.b.item(), 3), round(model.loss(x_train, y_train).item(), 3)))
print("done")
plt.show()
