import csv
import torch
import matplotlib.pyplot as plt

# Read data from file
day = []
head_circumfereance = []
with open('./data/day_head_circumference.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = -1
    for row in csv_reader:
        if line_count == -1:
            print(f'{", ".join(row)}')
            line_count += 1
        else:
            day.append(float(row[0]))
            head_circumfereance.append(float(row[1]))
            line_count += 1
    print(f'Processing {line_count} entries...')

# Observed/training input and output
x_train = torch.tensor(day).reshape(-1, 1)
y_train = torch.tensor(head_circumfereance).reshape(-1, 1)


class SigmoidRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = SigmoidRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
lr = 0.0000005
epoch_range = 100

optimizer = torch.optim.SGD([model.b, model.W], lr)
for epoch in range(epoch_range):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Visualize result
plt.plot(x_train, y_train, '.', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
plt.plot(x_train, model.f(x_train).detach(), '.', label='$y = f(x) = 20 * torch.sigmoid(x @ self.W + self.b) + 31$')
plt.legend()

# Show model variables and loss
plt.ylabel('head circumference')
plt.xlabel("days\n\nW = %s\nb = %s\nloss = %s\nepochs = %s\nlearning rate = %s"
           % (model.W.item(), model.b.item(), model.loss(x_train, y_train), epoch_range, lr))
plt.show()
print("Done")
