import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
from needle.data import MNISTDataset, DataLoader

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    main_ = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(),
                              nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    sum_ = nn.Residual(main_)
    return nn.Sequential(sum_, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = []
    modules.append(nn.Flatten())
    modules.append(nn.Linear(dim, hidden_dim))
    modules.append(nn.ReLU())

    for _ in range(num_blocks):
      modules.append(ResidualBlock(dim=hidden_dim, 
                                     hidden_dim=hidden_dim // 2, 
                                     norm=norm, 
                                     drop_prob=drop_prob))
    
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
      model.train()
    else:
      model.eval()

    total_loss = list()
    total_err = 0
    total_samples = 0
    loss_fn = nn.SoftmaxLoss()

    for X, y in dataloader:
      logits = model(X)
      loss = loss_fn(logits, y)
      
      total_samples += X.shape[0]
      preds = logits.numpy().argmax(axis=1)
      total_err += np.sum(preds != y.numpy())

      total_loss.append(loss.numpy())

      if opt:
        loss.backward()
        opt.step()

    avg_loss = np.mean(total_loss)
    error_rate = total_err / total_samples

    return error_rate, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    test_dataset = MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz",
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    model = MLPResNet(dim=784, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for e in range(epochs):
      train_err, train_loss = epoch(train_dataloader, model, opt)

    test_err, test_loss = epoch(test_dataloader, model)

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
