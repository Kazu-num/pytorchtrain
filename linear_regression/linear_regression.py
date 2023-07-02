import torch
from torch import nn, optim
from matplotlib import pyplot

# y = 1 + 2 x' + 3  x'' (true)
# y = w + w'x' + w''x'' (model)
w_true = torch.Tensor([1, 2, 3])

epoc_num = 100
# data
x = torch.cat([torch.ones(epoc_num, 1),torch.randn(epoc_num, 2)], dim=1)    
y = torch.mv(x, w_true) + torch.randn(epoc_num) * 0.5

def linear_scratch():

    w = torch.randn(3, requires_grad=True)

    gamma = 0.1
    losses = []
    for epoc in range(epoc_num):
        w.grad = None
        y_pred = torch.mv(x, w)

        # MSE
        loss = torch.mean((y - y_pred)**2)
        loss.backward()

        w.data = w.data - gamma * w.grad.data
        losses.append(loss.item())

    print(w)
    return losses

def linear_use_module():
    net = nn.Linear(in_features=3, out_features=1, bias=False)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    epoc_num = 100
    losses = []
    for epoc in range(epoc_num):
        optimizer.zero_grad()
        y_pred = net(x)
        loss = loss_fn(y_pred.view_as(y), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(list(net.parameters()))
    return losses

def prot_png(losses0, losses1):
    fig, ax = pyplot.subplots()
    ax.plot(losses0, color='blue', label='linear_scratch')
    ax.plot(losses1, color='orange', label='linear_use_module')
    ax.set_xlabel('epoc')
    ax.set_ylabel('loss')
    ax.legend()
    pyplot.savefig('linear_regression.png')

if __name__ == '__main__':
    losses0 = linear_scratch()
    losses1 = linear_use_module()
    prot_png(losses0, losses1)