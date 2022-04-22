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
y_train = torch.tensor([[1.0],  # Output
                        [1.0],
                        [1.0],
                        [0.0]])

# Initial values of model
W_init = [[0.0], [0.0]]
b_init = [[0.0]]


# Creating model
class SigmoidNANDModel:
    def __init__(self):
        self.W = torch.tensor(W_init, requires_grad=True)
        self.b = torch.tensor(b_init, requires_grad=True)

    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = SigmoidNANDModel()

# Optimizing model
optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Visualize result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Input
ax.scatter([0, 1, 0, 1], [0, 0, 1, 1], y_train.flatten(0).tolist(), 'o', label='Input')
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
fig.text(0.02, 0.80, "W = [%.2f, %.2f]\n"
                     "b = %.2f\n"
                     "loss = %.3f\n"
                     "epoch_range = %s\n"
                     "learning_rate = %s"
         % (model.W.tolist()[0][0], model.W.tolist()[1][0],
            model.b.item(),
            model.loss(x_train, y_train).item(),
            epochs,
            learning_rate))
plt.savefig("images/b_plot")
plt.show()
