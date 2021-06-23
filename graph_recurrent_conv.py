from typing import Union, Tuple, Callable
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size

import torch
from torch import Tensor
from torch.nn import Parameter, Tanh, Linear, RNNCell
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import reset, uniform, zeros

class DoubleRNNConv(MessagePassing):
    """
    Message passing network for recurrent graph convolutions.
    Parameters:
    - channels: size of the hidden state embeddings, should be equal to the hidden states defined for the RNN Cell.
    - rnn: predefined RNN Cell to operate (could be other: LSTM, GRU etc.).
    - aggr: way to combine messages from neighbors to the corresponding single node.
    - root_weight: boolean to learn weights for the node features. (theta)
    - root_bias: boolean to learn biases for the node features. (theta)
    - **kwargs: check MessagePassing class.
    """
    def __init__(self, channels: int, rnn: Callable, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super(DoubleRNNConv, self).__init__(aggr=aggr, **kwargs)

        self.channels = channels
        self.rnn = rnn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(channels, channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rnn)
        if self.root is not None:
            uniform(self.root.size(0), self.root)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            out += torch.matmul(x_r, self.root)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # Creating pair messages from RNN cell.
        h_i = self.rnn(edge_attr,x_i)
        h_j = self.rnn(edge_attr,x_j)
        next_msg = h_i * h_j
        return next_msg

    def __repr__(self):
        return '{}(In-Out: {})'.format(self.__class__.__name__, self.channels)