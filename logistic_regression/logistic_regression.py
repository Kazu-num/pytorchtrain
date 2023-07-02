import torch
from torch import nn, optim
from matplotlib import pyplot
from sklearn.datasets import load_iris

def logistic():
    # data (iris)
    iris = load_iris()
    x = iris.data[:100]
    y = iris.target[:100]
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    net = nn.Linear(4, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.25)

    losses = []
    epoc_num = 100
    for epoc in range(epoc_num):
        optimizer.zero_grad()
        y_pred = net(x)
        loss = loss_fn(y_pred.view_as(y), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    pyplot.plot(losses)
    pyplot.savefig('logistic_regression.png')

    h = net(x)
    prob = nn.functional.sigmoid(h)

    y_pred = prob > 0.5

    print((y.byte() == y_pred.view_as(y)).sum().item())

if __name__ == '__main__':
    logistic()
