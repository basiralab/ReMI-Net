import torch
import numpy as np
from torch.nn import Module, Sequential, RNNCell, LayerNorm, Sigmoid
import matplotlib.pyplot as plt

import numbers
from torch.nn.parameter import Parameter
from torch import Tensor, Size
from typing import Union, List

from graph_recurrent_conv import DoubleRNNConv

# Check device. You can manually change this line to use cpu only, do not forget to change it in all other files.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MinMaxLayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(MinMaxLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            self.weight = Parameter(torch.ones(self.weight.size()))
            self.bias = Parameter(torch.zeros(self.bias.size()))

    def forward(self, X: Tensor, dim:int = -1) -> Tensor:
        if dim == -1: 
            # Standard use of minmax normalization. You can also try it with the learnable parameters.
            return ( X - X.min() ) / ( X.max() - X.min() ) #* self.weight + self.bias
        elif dim == 0:
            mins, argmins = X.min(dim)
            maxs, argmaxs = X.max(dim)
            return ( ( X - mins ) / ( maxs - mins ) ) * self.weight + self.bias
        elif dim == 1: 
            mins, argmins = X.min(dim)
            maxs, argmaxs = X.max(dim)
            return ( ( X.transpose(1,0) - mins ) / ( maxs - mins ) ).transpose(1,0) * self.weight + self.bias

class RemiNet(Module):
    def __init__(self, n_nodes, input_size, hidden_sizes, recursion="cyclic",norm_method = "sigmoidnorm"):
        # We provide the current dimensionalities in comments to make it easy to follow. 
        super(RemiNet, self).__init__()

        # Ensure parameters are properly defined.
        assert recursion == "cyclic" or recursion == "vanilla"
        assert norm_method == "sigmoidnorm" or norm_method == "minmax" or norm_method == "no_norm"

        self.recursion = recursion
        self.norm_method = norm_method

        if self.norm_method == "minmax": self.minmax_norm_layer = MinMaxLayerNorm([n_nodes*n_nodes,input_size]) # (1225, 4) or (35*35, 4)
        if self.norm_method == "sigmoidnorm": self.signorm_layer = Sequential(LayerNorm([n_nodes*n_nodes,input_size]), Sigmoid()) # (1225, 4) or (35*35, 4)

        # Define 3 ReMI-Net Layers.
        rnn = RNNCell(input_size, hidden_sizes[0]) # (4, 12)
        self.rec_conv1 = DoubleRNNConv(hidden_sizes[0], rnn)

        rnn = RNNCell(input_size, hidden_sizes[1]) # (4, 36)
        self.rec_conv2 = DoubleRNNConv(hidden_sizes[1], rnn)

        rnn = RNNCell(input_size, hidden_sizes[2]) # (4, 24)
        self.rec_conv3 = DoubleRNNConv(hidden_sizes[2], rnn)

        self.hidden_sizes = hidden_sizes
        self.n_nodes = n_nodes

    def init_hidden(self,node_size,hidden_size):
        # Hidden layers are multi-featured node embeddings.
        return torch.zeros((node_size,hidden_size), device=device)

    def forward(self, data, time_points=2, cycles = 1, stop_point=0):
        # x shape = (35,1)
        # edge shape = (1225,4)
        cbts = []
        hid1 = self.init_hidden(self.n_nodes, self.hidden_sizes[0]) # (35,12)
        hid2 = self.init_hidden(self.n_nodes, self.hidden_sizes[1]) # (35,36)
        hid3 = self.init_hidden(self.n_nodes, self.hidden_sizes[2]) # (35,24)

        input_edge_attr = data[0].edge_attr
        input_edge_index = data[0].edge_index
        if self.norm_method == "sigmoidnorm": input_edge_attr = self.signorm_layer(input_edge_attr)
        if self.norm_method == "minmax": input_edge_attr = self.minmax_norm_layer(input_edge_attr)

        if self.recursion == "cyclic":
            # Cyclic Graph Recurrent Neural Network
            # The last output is at first time point.
            for t in range(time_points): 
                # Vanilla for the first cycle.
                # Update Hidden States
                hid1=self.rec_conv1(hid1, input_edge_index, input_edge_attr)
                hid2=self.rec_conv2(hid2, input_edge_index, input_edge_attr)
                hid3=self.rec_conv3(hid3, input_edge_index, input_edge_attr)
                # Combine outputs from each recurrent layer.
                out = torch.cat((hid1,hid2,hid3),dim=1)
                cbt = self.calculate_cbt(out)
                cbts.append(cbt)

            for c in range(cycles):
                # Enter the cycle.
                last_cycle = c == cycles - 1
                for t in range(time_points):
                    # If there are multiple cycles, all time points will be propagated in all cycles except the last.
                    hid1=self.rec_conv1(hid1, input_edge_index, input_edge_attr)
                    hid2=self.rec_conv2(hid2, input_edge_index, input_edge_attr)
                    hid3=self.rec_conv3(hid3, input_edge_index, input_edge_attr)
                    # Combine Outputs from each recurrent layer.
                    out = torch.cat((hid1,hid2,hid3),dim=1)
                    cbts[t] = self.calculate_cbt(out)
                    # Last timepoint in the last cycle may differ.
                    if last_cycle and t == stop_point:
                        break

        if self.recursion == "vanilla":
            for t in range(time_points): 
                # Vanilla Graph Recurrent Neural Network
                # Update Hidden States
                hid1=self.rec_conv1(hid1, input_edge_index, input_edge_attr)
                hid2=self.rec_conv2(hid2, input_edge_index, input_edge_attr)
                hid3=self.rec_conv3(hid3, input_edge_index, input_edge_attr)
                # Combine Outputs from each recurrent layer.
                out = torch.cat((hid1,hid2,hid3),dim=1)
                cbt = self.calculate_cbt(out)
                cbts.append(cbt)

        # Returns all output CBTs.
        return torch.stack(cbts)

    # Utility function to derive a CBT from a hidden state matrix.
    def calculate_cbt(self,out):
        return torch.sum(torch.abs(out.repeat(self.n_nodes,1,1) - torch.transpose(out.repeat(self.n_nodes,1,1), 0, 1)), 2)

    # Utility function to generate a population center from a set of subject specific CBTs.
    def generate_cbt_median(self, test_data, t=2):
        # This operation is a post-training operation, so turn on the evaluation mode for the model.
        self.eval()
        all_cbts = []
        for data in test_data:
            # Post-training propagation.
            cbts = self.forward(data,time_points=t)
            all_cbts.append(cbts.cpu().detach().numpy())
        all_cbts = np.array(all_cbts)
        # Select the element-wise median to find most centered connectivities.
        cbt_medians = np.median(all_cbts, axis=0)
        # Back to the training mode.
        self.train()
        return torch.from_numpy(cbt_medians).to(device)

if __name__ == "__main__":
    from dataset import prepare_data
    from utils import cast_data

    dataset = cast_data(prepare_data())
    data = dataset[0][0]

    model = RemiNet(35,4,[12,36,24])
    cbts = model(data)