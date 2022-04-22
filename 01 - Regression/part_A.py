import csv
import torch
import matplotlib.pyplot as plt

# Read data from file
length = []
weight = []
with open('./data/length_weight.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = -1
    for row in csv_reader:
        if line_count == -1:
            print(f'{", ".join(row)}')
            line_count += 1
        else:
            length.append(float(row[0]))
            weight.append(float(row[1]))
            line_count += 1
    print(f'Processing {line_count} entries...')

# Observed/training input and output
x_train = torch.tensor(length).reshape(-1, 1)
y_train = torch.tensor(weight).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize
lr = 0.0001
epoch_range = 1000000
optimizer = torch.optim.SGD([model.b, model.W], lr)
for epoch in range(epoch_range):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Visualize result
plt.plot(x_train, y_train, '.', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label='$y = f(x) = xW+b$')
plt.legend()

# Show model variables and loss
plt.ylabel('weight')
plt.xlabel("length\nW = %s\nb = %s\nloss = %s\nepoch_range = %s\nlearning_rate = %s"
           % (model.W, model.b, model.loss(x_train, y_train), epoch_range, lr))
plt.show()
print("Done")
