import torch
import matplotlib.pyplot as plt

# Trainging values
learning_rate = 1
epochs = 10_000

# Training data
x_train = torch.tensor([[0.0, 0.0],  # Input
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0]])
y_train = torch.tensor([[0.0],  # Output
                        [1.0],
                        [1.0],
                        [0.0]])

# Initial values of model
W1_init = [[1.0, -1.0], [1.0, -1.0]]
W2_init = [[-1.0], [-1.0]]
b1_init = [[1.0, 1.0]]
b2_init = [[-1.0]]


# Creating model
class SigmoidXORModel:
    def __init__(self):
        self.W1 = torch.tensor(W1_init, requires_grad=True)
        self.W2 = torch.tensor(W2_init, requires_grad=True)
        self.b1 = torch.tensor(b1_init, requires_grad=True)
        self.b2 = torch.tensor(b2_init, requires_grad=True)

    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    def f2(self, x):
        return torch.sigmoid(x @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = SigmoidXORModel()

# Optimizing model
optimizer = torch.optim.SGD([model.b1, model.W1, model.b2, model.W2], learning_rate)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Visualize result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Input
ax.scatter([0, 1, 0, 1], [0, 0, 1, 1], y_train.tolist(), 'o', label='Input')
# Model
grid_spacing = 20
x1_grid, x2_grid = torch.meshgrid([torch.linspace(0, 1, grid_spacing), torch.linspace(0, 1, grid_spacing)])
y_grid = torch.empty([grid_spacing, grid_spacing])
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        y_grid[i, j] = model.f(torch.tensor([x1_grid[i, j], x2_grid[i, j]])).detach()
plot1_f = ax.plot_wireframe(x1_grid, x2_grid, y_grid, color="green", label="Model")
plt.legend()
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
fig.text(0.02, 0.75, "W1 = [[%.2f, %.2f], [%.2f, %.2f]]\n"
                     "W2 = [%.2f, %.2f]\n"
                     "b1 = [%.2f, %.2f]\n"
                     "b2 = %.2f\n"
                     "loss = %.3f\n"
                     "epoch_range = %s\n"
                     "learning_rate = %s"
         % (model.W1.tolist()[0][0], model.W1.tolist()[0][1], model.W1.tolist()[1][0], model.W1.tolist()[1][1],
            model.W2.tolist()[0][0], model.W2.tolist()[1][0],
            model.b1.tolist()[0][0], model.b1.tolist()[0][1],
            model.b2.item(),
            model.loss(x_train, y_train).item(),
            epochs,
            learning_rate))
plt.savefig("images/c_plot")
plt.show()
