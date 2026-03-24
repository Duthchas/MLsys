from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import (
    Module,
    ReLU,
    Dropout,
    Linear,
    RMSNorm1d,
)


def apply_rotary_pos_emb(x, *, base: float = 10000.0):
    """
    Apply Rotary Position Embedding (RoPE) to the last dimension of q/k.
    Expects shape (batch, seq_len, num_head, dim_head) with even dim_head.
    Matches HuggingFace LLaMA rotate_half convention.
    """
    batch, seq_len, num_head, dim_head = x.shape
    if dim_head % 2 != 0:
        # If head dimension is odd, skip rotation and return input unchanged.
        return x

    half_dim = dim_head // 2
    device = getattr(x, "device", None)
    dtype = getattr(x, "dtype", "float32")

    inv_freq = base ** (-np.arange(0, half_dim, dtype=np.float32) / half_dim)
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = positions[:, None] * inv_freq[None, :]

    cos = np.cos(freqs)
    sin = np.sin(freqs)

    # HuggingFace LLaMA: concatenate freqs with itself (block-wise repeat).
    cos_full = np.concatenate([cos, cos], axis=-1)
    sin_full = np.concatenate([sin, sin], axis=-1)

    cos_t = Tensor(cos_full, device=device, dtype=dtype, requires_grad=False)
    sin_t = Tensor(sin_full, device=device, dtype=dtype, requires_grad=False)

    cos_t = ops.reshape(cos_t, (1, seq_len, 1, dim_head))
    sin_t = ops.reshape(sin_t, (1, seq_len, 1, dim_head))

    cos_b = ops.broadcast_to(cos_t, x.shape)
    sin_b = ops.broadcast_to(sin_t, x.shape)

    # HF rotate_half: block-wise concat [-x2, x1]
    x_halves = ops.reshape(x, (batch, seq_len, num_head, 2, half_dim))
    x1, x2 = ops.split(x_halves, axis=3)
    x1 = ops.reshape(x1, (batch, seq_len, num_head, half_dim))
    x2 = ops.reshape(x2, (batch, seq_len, num_head, half_dim))

    rotated = ops.stack((-x2, x1), axis=3)  # shape (b, s, h, 2, half_dim)
    rotated = ops.reshape(rotated, x.shape)

    return x * cos_b + rotated * sin_b


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return mask

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        scale = np.sqrt(q_dim)

        logits = self.matmul(q, k)
        logits = ops.divide_scalar(logits, scale)

        if self.causal:
            mask = self.create_causal_mask(queries_len, keys_values_len, q.device)
            mask = Tensor(
                mask,
                device=q.device,
                dtype=q.dtype,
                requires_grad=False,
            )
            mask = mask.reshape((1, 1, queries_len, keys_values_len))
            mask = ops.broadcast_to(mask, logits.shape)
            logits = logits + mask

        probs = self.softmax(logits)
        probs = self.dropout(probs)

        v_t = ops.transpose(v, axes=(2, 3))
        result = self.matmul(probs, v_t)
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        num_kv_heads: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        rope_base: float = 10000.0,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_head
        self.rope_base = rope_base

        if self.num_head % self.num_kv_heads != 0:
            raise ValueError("num_head must be divisible by num_kv_heads for GQA.")

        self.prenorm_q = RMSNorm1d(q_features, device=device, dtype=dtype)
        self.prenorm_k = RMSNorm1d(k_features, device=device, dtype=dtype)
        self.prenorm_v = RMSNorm1d(v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        kv_inner_dim = self.num_kv_heads * dim_head

        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, kv_inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, kv_inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        batch = batch_size
        inner_dim = self.num_head * self.dim_head

        q_norm = self.prenorm_q(
            ops.reshape(q, (batch * queries_len, q_dim))
        )
        q_norm = ops.reshape(q_norm, (batch, queries_len, q_dim))

        k_norm = self.prenorm_k(
            ops.reshape(k, (batch * keys_values_len, k_dim))
        )
        k_norm = ops.reshape(k_norm, (batch, keys_values_len, k_dim))

        v_norm = self.prenorm_v(
            ops.reshape(v, (batch * keys_values_len, v_dim))
        )
        v_norm = ops.reshape(v_norm, (batch, keys_values_len, v_dim))

        q_proj = self.q_projection(
            ops.reshape(q_norm, (batch * queries_len, q_dim))
        )
        q_proj = ops.reshape(q_proj, (batch, queries_len, self.num_head, self.dim_head))
        q_proj = apply_rotary_pos_emb(q_proj, base=self.rope_base)
        q_proj = ops.transpose(q_proj, axes=(1, 2))

        k_proj = self.k_projection(
            ops.reshape(k_norm, (batch * keys_values_len, k_dim))
        )
        k_proj = ops.reshape(k_proj, (batch, keys_values_len, self.num_kv_heads, self.dim_head))
        k_proj = apply_rotary_pos_emb(k_proj, base=self.rope_base)
        k_proj = ops.transpose(k_proj, axes=(1, 2))  # (batch, kv_heads, kv_len, dim_head)
        if self.num_kv_heads != self.num_head:
            repeat = self.num_head // self.num_kv_heads
            k_proj = ops.reshape(k_proj, (batch, self.num_kv_heads, 1, keys_values_len, self.dim_head))
            k_proj = ops.broadcast_to(k_proj, (batch, self.num_kv_heads, repeat, keys_values_len, self.dim_head))
            k_proj = ops.reshape(k_proj, (batch, self.num_head, keys_values_len, self.dim_head))
        else:
            k_proj = ops.reshape(k_proj, (batch, self.num_head, keys_values_len, self.dim_head))

        v_proj = self.v_projection(
            ops.reshape(v_norm, (batch * keys_values_len, v_dim))
        )
        v_proj = ops.reshape(v_proj, (batch, keys_values_len, self.num_kv_heads, self.dim_head))
        v_proj = ops.transpose(v_proj, axes=(1, 2))  # (batch, kv_heads, kv_len, dim_head)
        if self.num_kv_heads != self.num_head:
            repeat = self.num_head // self.num_kv_heads
            v_proj = ops.reshape(v_proj, (batch, self.num_kv_heads, 1, keys_values_len, self.dim_head))
            v_proj = ops.broadcast_to(v_proj, (batch, self.num_kv_heads, repeat, keys_values_len, self.dim_head))
            v_proj = ops.reshape(v_proj, (batch, self.num_head, keys_values_len, self.dim_head))
        else:
            v_proj = ops.reshape(v_proj, (batch, self.num_head, keys_values_len, self.dim_head))

        attn_out, probs = self.attn(q_proj, k_proj, v_proj)
        self.probs = probs

        attn_out = ops.transpose(attn_out, axes=(1, 2))
        attn_out = ops.reshape(attn_out, (batch * queries_len, inner_dim))

        result = self.out_projection(attn_out)
        result = ops.reshape(result, (batch, queries_len, self.out_features))
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size

        ### BEGIN YOUR SOLUTION
        self.attention = AttentionLayer(
            q_features,
            num_head,
            dim_head,
            k_features=q_features,
            v_features=q_features,
            out_features=q_features,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype,
        )

        self.attn_dropout = Dropout(dropout)

        self.ff_norm = RMSNorm1d(q_features, device=device, dtype=dtype)
        self.ff_linear1 = Linear(q_features, 2 * hidden_size, device=device, dtype=dtype)
        self.ff_dropout = Dropout(dropout)
        self.ff_linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.ff_output_dropout = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        attn_out = self.attention(x)
        attn_out = self.attn_dropout(attn_out)
        x = x + attn_out

        norm_in = ops.reshape(x, (batch_size * seq_len, x_dim))
        normed = self.ff_norm(norm_in)

        ff1 = self.ff_linear1(normed)
        ff1 = ops.reshape(ff1, (batch_size, seq_len, 2, self.hidden_size))
        u, v = ops.split(ff1, axis=2)
        u = ops.reshape(u, (batch_size, seq_len, self.hidden_size))
        v = ops.reshape(v, (batch_size, seq_len, self.hidden_size))

        # Swish(v) = v * sigmoid(v)
        exp_neg_v = ops.exp(ops.negate(v))
        denom = ops.add_scalar(exp_neg_v, 1.0)
        sigmoid_v = ops.power_scalar(denom, -1)
        swish_v = v * sigmoid_v

        hidden = u * swish_v
        hidden = ops.reshape(hidden, (batch_size * seq_len, self.hidden_size))
        hidden = self.ff_dropout(hidden)
        hidden = self.ff_linear2(hidden)
        hidden = self.ff_output_dropout(hidden)

        hidden = ops.reshape(hidden, (batch_size, seq_len, x_dim))
        x = x + hidden
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.sequence_len = sequence_len

        self.layers: List[TransformerLayer] = []
        for idx in range(num_layers):
            layer = TransformerLayer(
                embedding_size,
                num_head,
                dim_head,
                hidden_size,
                dropout=dropout,
                causal=causal,
                device=device,
                dtype=dtype,
            )
            self.layers.append(layer)
            setattr(self, f"layer_{idx}", layer)
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, embed_dim = x.shape

        for layer in self.layers:
            x = layer(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
