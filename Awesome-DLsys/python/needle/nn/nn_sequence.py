"""The module.
"""
from typing import List
import math
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        exp_x = ops.exp(x)
        return ops.divide(exp_x, ops.add_scalar(exp_x, 1.0))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        assert nonlinearity in ['tanh', 'relu']
        self.nonlinearity = nonlinearity

        bound = math.sqrt(1 / hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)

        linear = ops.matmul(X, self.W_ih)
        if self.bias_ih is not None:
            linear = linear + ops.broadcast_to(ops.reshape(self.bias_ih, (1, self.hidden_size)), linear.shape)

        hidden_term = ops.matmul(h, self.W_hh)
        if self.bias_hh is not None:
            hidden_term = hidden_term + ops.broadcast_to(ops.reshape(self.bias_hh, (1, self.hidden_size)), hidden_term.shape)

        pre_act = linear + hidden_term

        if self.nonlinearity == 'tanh':
            return ops.tanh(pre_act)
        else:
            return ops.relu(pre_act)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        self.rnn_cells: List[RNNCell] = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cell = RNNCell(
                in_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            self.rnn_cells.append(cell)
            setattr(self, f"rnn_cell_{layer}", cell)
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len = X.shape[0]
        bs = X.shape[1]

        if h0 is None:
            hidden_states = [
                init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h0_splits = ops.split(h0, axis=0)
            hidden_states = [
                ops.reshape(h_part, (bs, self.hidden_size))
                for h_part in h0_splits
            ]

        outputs: List[Tensor] = []
        x_slices = ops.split(X, axis=0)
        for t in range(seq_len):
            x_t = ops.reshape(x_slices[t], (bs, -1))
            input_t = x_t
            new_hidden_states = []
            for layer, cell in enumerate(self.rnn_cells):
                h_prev = hidden_states[layer]
                h_new = cell(input_t, h_prev)
                new_hidden_states.append(h_new)
                input_t = h_new
            hidden_states = new_hidden_states
            outputs.append(input_t)

        output_tensor = ops.stack(outputs, axis=0)
        h_n = ops.stack(hidden_states, axis=0)
        return output_tensor, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        bound = math.sqrt(1 / hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h_prev = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c_prev = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h_prev, c_prev = h

        gates = ops.matmul(X, self.W_ih)
        if self.bias_ih is not None:
            gates = gates + ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4 * self.hidden_size)), gates.shape)

        gates = gates + ops.matmul(h_prev, self.W_hh)
        if self.bias_hh is not None:
            gates = gates + ops.broadcast_to(ops.reshape(self.bias_hh, (1, 4 * self.hidden_size)), gates.shape)

        gates = ops.reshape(gates, (bs, 4, self.hidden_size))
        gate_splits = ops.split(gates, axis=1)

        def squeeze_gate(gate_tensor):
            return ops.reshape(gate_tensor, (bs, self.hidden_size))

        sigmoid = Sigmoid()
        i = sigmoid.forward(squeeze_gate(gate_splits[0]))
        f = sigmoid.forward(squeeze_gate(gate_splits[1]))
        g = ops.tanh(squeeze_gate(gate_splits[2]))
        o = sigmoid.forward(squeeze_gate(gate_splits[3]))

        c_prime = f * c_prev + i * g
        h_prime = o * ops.tanh(c_prime)
        return h_prime, c_prime
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.lstm_cells: List[LSTMCell] = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cell = LSTMCell(
                in_size,
                hidden_size,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            self.lstm_cells.append(cell)
            setattr(self, f"lstm_cell_{layer}", cell)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len = X.shape[0]
        bs = X.shape[1]

        if h is None:
            hidden_states = [
                init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
                for _ in range(self.num_layers)
            ]
            cell_states = [
                init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h0, c0 = h
            h_splits = ops.split(h0, axis=0)
            c_splits = ops.split(c0, axis=0)
            hidden_states = [ops.reshape(h_part, (bs, self.hidden_size)) for h_part in h_splits]
            cell_states = [ops.reshape(c_part, (bs, self.hidden_size)) for c_part in c_splits]

        outputs: List[Tensor] = []
        x_slices = ops.split(X, axis=0)
        for t in range(seq_len):
            x_t = ops.reshape(x_slices[t], (bs, -1))
            input_t = x_t
            new_hidden_states = []
            new_cell_states = []
            for layer, cell in enumerate(self.lstm_cells):
                h_prev = hidden_states[layer]
                c_prev = cell_states[layer]
                h_new, c_new = cell(input_t, (h_prev, c_prev))
                new_hidden_states.append(h_new)
                new_cell_states.append(c_new)
                input_t = h_new
            hidden_states = new_hidden_states
            cell_states = new_cell_states
            outputs.append(input_t)

        output_tensor = ops.stack(outputs, axis=0)
        h_n = ops.stack(hidden_states, axis=0)
        c_n = ops.stack(cell_states, axis=0)
        return output_tensor, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        total = seq_len * bs
        flat = ops.reshape(x, (total,))
        one_hots = []
        for idx_tensor in ops.split(flat, axis=0):
            idx_scalar = ops.reshape(idx_tensor, ())
            one_hots.append(
                init.one_hot(
                    self.num_embeddings,
                    idx_scalar,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        one_hot_matrix = ops.stack(one_hots, axis=0)
        embeddings_flat = ops.matmul(one_hot_matrix, self.weight)
        embeddings = ops.reshape(embeddings_flat, (seq_len, bs, self.embedding_dim))
        return embeddings
        ### END YOUR SOLUTION
