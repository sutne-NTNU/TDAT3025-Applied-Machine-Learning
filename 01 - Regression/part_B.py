import csv
import torch
import matplotlib.pyplot as plt

# Training values
lr = 0.0001
epoch_range = 1_000_000

# Read data from file
length = []
weight = []
length_weight = []
days = []
with open('./data/day_length_weight.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = -1
    for row in csv_reader:
        if line_count == -1:
            print(f'{", ".join(row)}')
            line_count += 1
        else:
            length.append(float(row[1]))
            weight.append(float(row[2]))
            length_weight.append([float(row[1]), float(row[2])])
            days.append(float(row[0]))
            line_count += 1
    print(f'Processing {line_count} entries.')

# Observed input and output
x_train = torch.tensor([length_weight]).reshape(-1, 2)
y_train = torch.tensor(days).reshape(-1, 1)


# create model
class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize model
optimizer = torch.optim.SGD([model.b, model.W], lr)
for epoch in range(epoch_range):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Visualize result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Input
ax.scatter3D(length, weight, days, c='b', marker='.', label="Input")
# Model
grid_spacing = 10
x1 = torch.linspace(min(length), max(length), grid_spacing)
x2 = torch.linspace(min(weight), max(weight), grid_spacing)
x1_grid, x2_grid = torch.meshgrid(x1, x2)
y_grid = torch.empty([grid_spacing, grid_spacing])
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        y_grid[i, j] = model.f(torch.tensor([x1_grid[i, j], x2_grid[i, j]])).detach()
plot1_f = ax.plot_wireframe(x1_grid, x2_grid, y_grid, color="green", label="Predictor")
plt.legend()
ax.set_xlabel('Length\n')
ax.set_ylabel('Weight')
ax.set_zlabel('  Days')
fig.text(0.02, 0.80, "W = [%.2f, %.2f]\n"
                     "b = %.2f\n"
                     "loss = %.3f\n"
                     "epoch_range = %s\n"
                     "learning_rate = %s"
         % (model.W.tolist()[0][0], model.W.tolist()[1][0],
            model.b.item(),
            model.loss(x_train, y_train).item(),
            epoch_range,
            lr))
plt.savefig("images/b_plot")
plt.show()
