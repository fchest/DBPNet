import math
import torch.nn as nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        """
        Multi-headed attention. This module can use the MULTIHEADATTENTION module built in Pytorch1.9.
        @param embed_dim: input embedding
        @param num_heads: number of heads
        @param attn_dropout: dropout applied on the attention weights
        @param bias: whether to add bias to q
        @param add_bias_kv: whether to add bias to kv
        @param add_zero_attn: add a new batch of zeros to the key and value sequences at dim=1.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """
        @param query: (Time, Batch, Channel)
        @param key: (Time, Batch, Channel)
        @param value: (Time, Batch, Channel)
        @param attn_mask: mask that prevents attention to certain positions.
        @return: a tuple (output, weight), output shape (Time, Batch, Channel)
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights / k.shape[1] ** 0.5
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights = attn_weights + attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 embed_dropout=0.0, attn_mask=False):
        """
        Transformer encoder consisting of N layers. Each layer is a TransformerEncoderLayer.
        @param embed_dim: input embedding
        @param num_heads: number of heads
        @param layers: number of layers
        @param attn_dropout: dropout applied on the attention weights
        @param relu_dropout: dropout applied on the first layer of the residual block
        @param res_dropout: dropout applied on the residual block
        @param embed_dropout: dropout applied on the residual block
        @param attn_mask: whether to apply mask on the attention weights
        """
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.attn_mask = attn_mask

        self.positionencoding = PositionalEncoding(embed_dim, embed_dropout)

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_in):
        """
        @param x_in: embedded input of shape (src_len, batch, embed_dim)
        @param return_: whether to return the weight list
        @return: the last encoder layer's output of shape (src_len, batch, embed_dim).
            if return_=True, return tuple (output, weights)
        """
        # embed tokens
        x = self.positionencoding(self.embed_scale * x_in)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        """
        Encoder layer block
        @param embed_dim: input embedding
        @param num_heads: number of heads
        @param attn_dropout: dropout applied on the attention weights
        @param relu_dropout: dropout applied on the first layer of the residual block
        @param res_dropout: dropout applied on the residual block
        @param attn_mask: whether to apply mask on the attention weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask
        self.attn_weights = None

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        @param x: (seq_len, batch, embed_dim)
        @param x_k: (seq_len, batch, embed_dim)
        @param x_v: (seq_len, batch, embed_dim)
        @param return_: whether to return the weight list
        @return: encoded output of shape (batch, src_len, embed_dim).
            if return_=True, return tuple (output, weight)
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        self.attn_weights = _
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        # Position-wise feed forward
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(torch.device('cuda:0'))
    return future_mask[:dim1, :dim2]

#@save
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class TABNet(nn.Module):
    def __init__(self, in_channels):
        super(TABNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(64, 16)
        self.linear2 = nn.Linear(16, 4)
        self.cm_attn = TransformerEncoder(
            embed_dim=in_channels,
            num_heads=1,
            layers=1,
            attn_dropout=0,
            relu_dropout=0,
            res_dropout=0,
            embed_dropout=0,
            attn_mask=False
        )

    def forward(self, eeg):
        x = torch.squeeze(eeg, dim=1)
        x = x.permute(2, 0, 1)
        x = self.cm_attn(x)
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear2(x)
        x = F.relu(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, (1, kernel_size, kernel_size), (1, stride, stride), (0, padding, padding), bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, num_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, num_channels, 1, stride=1)
        self.conv2 = ConvLayer(num_channels, num_channels, 3, stride=stride, padding=1)
        self.conv3 = ConvLayer(num_channels, out_channels, 1, stride = 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ConvLayer(in_channels, out_channels, 1, stride=stride)
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        x = self.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return x

class FRBNet(nn.Module):
    def __init__(self):
        super(FRBNet, self).__init__()
        self.conv1 = ConvLayer(1, 32, 7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ResBlock(32, 32, 64),
        )
        self.conv2 = nn.Conv3d(64, 64, 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm3d(64)
        self.layer2 = nn.Sequential(
            ResBlock(64, 64, 128, stride=2),
        )
        self.conv3 = nn.Conv3d(128, 128, 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm3d(128)
        self.layer3 = nn.Sequential(
            ResBlock(128, 128, 256, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv5 = nn.Conv3d(256, 4, 1)
        self.bn5 = nn.BatchNorm3d(4)
        self.linear = nn.Linear(8, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = F.tanh(x)
        x = x.reshape(x.shape[0], -1, x.shape[3],  x.shape[4])
        x = self.pool1(x)
        x = x.reshape(x.shape[0], 32, -1, x.shape[2], x.shape[3])
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        return x