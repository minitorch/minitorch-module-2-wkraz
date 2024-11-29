"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import sys
import os
import numpy as np

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # Follow the original approach for forward propagation
        y1 = self.layer1.forward(x).relu()
        y2 = self.layer2.forward(y1).relu()
        y3 = self.layer3.forward(y2).sigmoid()
        return y3


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch_size, in_size = x.shape

        # Match the original logic for the weighted sum and bias addition
        w = self.weights.value.view(1, in_size, self.out_size)
        x = x.view(batch_size, in_size, 1)
        t = w * x  # Shape: (batch_size, in_size, out_size)
        t = t.sum(1).view(batch_size, self.out_size)
        b = self.bias.value.view(1, self.out_size)
        return t + b


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)  # Ensure data is correctly converted
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward pass
            out = self.model.forward(X).view(len(data.X))
            # was having issues as this was int not tensor, so change to tensor
            one_tensor = minitorch.tensor([1.0])  
            prob = (out * y) + ((one_tensor - out) * (one_tensor - y))

            # Loss calculation
            loss = -prob.log().sum()
            loss.backward()  # Backpropagation
            total_loss = loss.item()
            losses.append(total_loss)

            # Update parameters
            optim.step()

            # Logging every 10 epochs or on the final epoch
            if epoch % 10 == 0 or epoch == max_epochs:
                correct = int(((out.detach() > 0.5) == y).sum().item())
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.1
    # Iterate over all available datasets
    for dataset_name in minitorch.datasets.keys():
        print(f"Training on dataset: {dataset_name}")
        data = minitorch.datasets[dataset_name](PTS)
        TensorTrain(HIDDEN).train(data, RATE)