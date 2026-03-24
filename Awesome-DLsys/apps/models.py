import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device = device
        self.dtype = dtype

        def conv_bn(in_channels, out_channels, kernel_size, stride):
            return nn.Sequential(
                nn.Conv(in_channels, out_channels, kernel_size, stride=stride, device=device, dtype=dtype),
                nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
                nn.ReLU(),
            )

        self.block1 = conv_bn(3, 16, 7, 4)
        self.block2 = conv_bn(16, 32, 3, 2)

        self.block3 = conv_bn(32, 32, 3, 1)
        self.block4 = conv_bn(32, 32, 3, 1)

        self.block5 = conv_bn(32, 64, 3, 2)
        self.block6 = conv_bn(64, 128, 3, 2)
        self.block7 = conv_bn(128, 128, 3, 1)
        self.block8 = conv_bn(128, 128, 3, 1)

        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        out = self.block1(x)
        out = self.block2(out)

        residual = out
        out = self.block3(out)
        out = self.block4(out)
        out = out + residual

        out = self.block5(out)
        out = self.block6(out)

        residual = out
        out = self.block7(out)
        out = self.block8(out)
        out = out + residual

        N = out.shape[0]
        out = ndl.ops.reshape(out, (N, -1))
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.seq_model_type = seq_model
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.sequence = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, bias=True, nonlinearity='tanh', device=device, dtype=dtype)
        else:
            self.sequence = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bias=True, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        embeddings = self.embedding(x)
        outputs, h_next = self.sequence(embeddings, h)
        seq_len, bs, _ = outputs.shape
        logits = self.linear(ndl.ops.reshape(outputs, (seq_len * bs, self.hidden_size)))
        return logits, h_next
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
