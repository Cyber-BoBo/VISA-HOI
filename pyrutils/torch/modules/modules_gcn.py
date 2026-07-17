import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.utils.checkpoint as cp

from torch.nn.parameter import Parameter



from typing import Optional, Sequence, Union
from pyrutils.torch.general import pick_activation_function, MySequential


class GeometricGraph(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        node_n (int): Number of joints in the human body    关节点的数量
        in_channels (int): Number of channels in the input sequence data        输入通道数
        embed_size (int): Number of channels produced by the convolution      输出通道数
    """
    #----------------------------------------------------
    """ 初始化 """
    def __init__(self,
                 node,
                 in_channels,
                 embed_size,
                 delta
                 ):
        super(GeometricGraph, self).__init__()

        self.s_gcn_node = node
        self.data_dims = in_channels
        self.embed_size = embed_size
        self.delta = delta

        self.joint_embed = embed(self.data_dims, 64, self.s_gcn_node, norm=True, bias=True)
        self.get_sim = compute_similarity(64,128,bias= True)
        
        # self.gcn = build_gcn([64, self.embed_size], ['relu'])       #1
        self.gcn = build_gcn([64, 128, self.embed_size], ['relu', 'relu'])       #2
        # self.gcn = build_gcn([64, 64, 128, self.embed_size], ['relu', 'relu', 'relu'])       #3
        # self.gcn = build_gcn([64, 64, 128,128, self.embed_size], ['relu', 'relu', 'relu','relu'])       #4


    def forward(self, x):
        x = self.joint_embed(x)
        sim = self.get_sim(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x_out, _ = self.gcn(x, sim)
        if not self.delta:
            x_out = x_out.permute(0, 3, 2, 1).contiguous()
        return x_out

def build_gcn(dims: Sequence[int], activations: Optional[Sequence[Union[str, dict]]] = None,
              dropout: float = 0.0, bias: bool = True):
    """
    Build a gcn.
    """
    if activations is None:
        activations = ['identity'] * (len(dims) - 1)
    if len(dims) - 1 != len(activations):
        raise ValueError('Number of activations must be the same as the number of dimensions - 1.')
    layers = []
    for dim_in, dim_out, activation in zip(dims[:-1], dims[1:], activations):
        layers.append(GraphConvolutionLayer(dim_in, dim_out, activation, bias=bias, dropout=dropout))
    return MySequential(*layers)

class GraphConvolutionLayer(nn.Module):
    def __init__(self, dim_in, dim_out, activation = 'relu', bias = True, dropout = 0.0):
        super(GraphConvolutionLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = Parameter(torch.FloatTensor(dim_in, dim_out))
        self.activation = pick_activation_function(activation)
        if bias:
            self.bias = Parameter(torch.FloatTensor(dim_out))
        else:
            self.register_parameter('bias',None)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.register_parameter('dropout',None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
    
    def forward(self, input, adj):
        x = torch.matmul(input, self.weight)
        x = torch.matmul(adj,x)
        # x = cp.checkpoint(gcn_fn, input, use_reentrant=False)
        # x = torch.spmm(adj,x)
        if self.bias is not None:
            x = x + self.bias
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x, adj



""" Geometric GCN """
class Geo_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        node_n (int): Number of joints in the human body    关节点的数量
        in_channels (int): Number of channels in the input sequence data        输入通道数
        out_channels (int): Number of channels produced by the convolution      输出通道数
    """

    #----------------------------------------------------
    """ 初始化 """
    def __init__(self,
                 node_n,
                 in_channels,
                 out_channels):
        super(Geo_gcn, self).__init__()

        self.joint_embed = embed(in_channels, 64, node_n, norm=True, bias=True)
        self.get_s = compute_similarity(64, 128, bias = True)
        self.weight = Parameter(torch.FloatTensor(64, out_channels))
        self.reset_parameters()
        
    # regularisation
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.joint_embed(x)     # 8 64 26 272
        s = self.get_s(x)           # 8
        x = x.permute(0, 3, 2, 1).contiguous()
        x = s.matmul(x)
        x = torch.matmul(x, self.weight)           # 矩阵相乘
        x = x.permute(0, 3, 2, 1).contiguous()
        return x

""" 数据标准化 """
class norm_data(nn.Module):
    def __init__(self, dim= 64, node_n = 19):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim*node_n)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 4, dim1 = 128, node_n = 19, norm = True, bias = False):
        """
        dim- 输入维度
        dim1- 输出维度
        node_n- 人体关节点的数量
        norm- 是否标准化
        bias- 是否使用偏置
        """
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim, node_n),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

""" 1x1的卷积层(2维卷积) """
class cnn1x1(nn.Module):
    def __init__(self, dim1 = 4, dim2 =4, bias = True):

        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

""" 计算相似性 """
class compute_similarity(nn.Module):
    def __init__(self, dim1, dim2, bias = False):
        super(compute_similarity, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.s1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.s2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)       # 对行进行Softmax运算

    def forward(self, x):
        s1 = self.s1(x).permute(0, 3, 2, 1).contiguous()
        s2 = self.s2(x).permute(0, 3, 1, 2).contiguous()
        s3 = s1.matmul(s2)
        s = self.softmax(s3)
        return s
