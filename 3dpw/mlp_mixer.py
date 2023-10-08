import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import utils.tools as tools
import numpy as np


class Graph():
    def __init__(self, node_n, seq_len):
        self.node_num = node_n
        self.seq_len = seq_len
        self.get_edge()

    def get_edge(self):
        self_link = [(i, i) for i in range(self.node_num)]
        bone_link = [[0, 2], [1, 2], [0, 3], [3, 6], [6, 9], [1, 4], [4, 7], [7, 10], [2, 5], [5, 8],
                     [8, 12],
                     [12, 15], [15, 17], [17, 19], [19, 21], [8, 13], [13, 16], [16, 18], [18, 20], [20, 22],
                     [8, 11], [11, 14]]
        self.bone_link = bone_link
        self.self_link = self_link
        A_ske_in = np.zeros((self.node_num, self.node_num))
        A_ske_out = np.zeros((self.node_num, self.node_num))
        A_ske_self = np.zeros((self.node_num, self.node_num))
        for i, j in self.bone_link:
            A_ske_in[i, j] = 1
            A_ske_out[j, i] = 1
        for i, j in self.self_link:
            A_ske_self[i, j] = 1
            A_ske_self[j, i] = 1
        A_ske_in = tools.normalize_digraph(A_ske_in)
        A_ske_out = tools.normalize_digraph(A_ske_out)
        self.A_ske = np.stack((A_ske_in, A_ske_out, A_ske_self))

        A_tem_self = np.zeros((self.seq_len, self.seq_len))
        A_tem_in = np.zeros((self.seq_len, self.seq_len))
        # A_tem_out = np.zeros((self.seq_len, self.seq_len))
        for i in range(self.seq_len - 1):
            A_tem_self[i, i] = 1
            A_tem_self[i + 1, i + 1] = 1
            A_tem_in[i, i + 1] = 1
            A_tem_in[i + 1, i] = 1
        A_tem_in = tools.normalize_digraph(A_tem_in)
        # A_tem_out = tools.normalize_digraph(A_tem_out)
        self.A_tem = np.stack((A_tem_in, A_tem_self))


class A_adp(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_c, out_c, node_n=22, seq_len_in=35, seq_len_out=70, bias=False):
        super(A_adp, self).__init__()
        self.in_features = in_c
        self.out_features = out_c
        self.conv_c1 = nn.Conv2d(in_c, out_c, 1)
        self.conv_c2 = nn.Conv2d(in_c, out_c, 1)
        # self.conv_t1 = nn.Conv2d(seq_len_in, seq_len_out, 1)
        # self.conv_t2 = nn.Conv2d(seq_len_in, seq_len_out, 1)
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.bn1 = nn.BatchNorm1d(node_n * node_n)
        self.act_f = nn.Softmax(-2)
        self.do = nn.Dropout(0.3)
        # self.conv=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        self.register_parameter('bias', None)
        # self.reset_parameters()

        self.support = None

    '''def reset_parameters(self):
        stdv = 1. / math.sqrt(self.seq_len)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
'''

    def forward(self, input):
        # input [b,c,17,35]
        b, c, n, l = input.shape
        # output1 = self.conv_c1(input)
        # output2 = self.conv_c2(input)
        output1 = self.conv_c1(input.permute(0, 1, 3, 2)).contiguous().view(b, self.out_features * l, n)
        output2 = self.conv_c2(input.permute(0, 1, 3, 2)).contiguous().view(b, self.out_features * l, n)
        # output1 = self.conv_t1(output1.permute(0, 3, 1, 2))  # [b,c,17,35] -> [b,35,c,17]
        # output2 = self.conv_t2(output2.permute(0, 3, 1, 2))  # [b,c,17,35] -> [b,35,c,17]
        output = torch.matmul(output2.permute(0, 2, 1), output1) / output1.size(-2)
        output = output.view(b, -1).contiguous()
        output = self.bn1(output).view(b, n, n).contiguous()
        A_adp = self.act_f(output)
        # a = A_adp.cpu().detach().numpy()
        A_adp = self.do(A_adp)
        # a = A_adp.cpu().detach().numpy()

        # A_adp = torch.mul(A_adp, self.M_adp)

        return A_adp.unsqueeze(1)


class TA_adp(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_c, out_c, node_n=22, seq_len_in=35, seq_len_out=70, bias=False):
        super(TA_adp, self).__init__()
        self.in_features = in_c
        self.out_features = out_c
        self.conv_c1 = nn.Conv2d(in_c, out_c, 1)
        self.conv_c2 = nn.Conv2d(in_c, out_c, 1)
        # self.conv_t1 = nn.Conv2d(seq_len_in, seq_len_out, 1)
        # self.conv_t2 = nn.Conv2d(seq_len_in, seq_len_out, 1)
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.bn1 = nn.BatchNorm1d(seq_len_in * seq_len_in)
        self.act_f = nn.Softmax(-2)
        self.do = nn.Dropout(0.3)
        # self.conv=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        self.register_parameter('bias', None)
        # self.reset_parameters()

        self.support = None

    '''def reset_parameters(self):
        stdv = 1. / math.sqrt(self.seq_len)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
'''

    def forward(self, input):
        # input [b,c,17,35]
        b, c, n, l = input.shape
        # output1 = self.conv_c1(input)
        # output2 = self.conv_c2(input)
        output1 = self.conv_c1(input.permute(0, 1, 3, 2)).contiguous().view(b, self.out_features * l, n)
        output2 = self.conv_c2(input.permute(0, 1, 3, 2)).contiguous().view(b, self.out_features * l, n)
        # output1 = self.conv_t1(output1.permute(0, 3, 1, 2))  # [b,c,17,35] -> [b,35,c,17]
        # output2 = self.conv_t2(output2.permute(0, 3, 1, 2))  # [b,c,17,35] -> [b,35,c,17]
        output = torch.matmul(output2.permute(0, 2, 1), output1) / output1.size(-2)
        output = output.view(b, -1).contiguous()
        output = self.bn1(output).view(b, n, n).contiguous()
        A_adp = self.act_f(output)
        # a = A_adp.cpu().detach().numpy()
        A_adp = self.do(A_adp)
        # a = A_adp.cpu().detach().numpy()

        # A_adp = torch.mul(A_adp, self.M_adp)

        return A_adp.unsqueeze(1)


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_c, out_c, node_n=22, seq_len=35, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_c
        self.out_features = out_c
        # self.att1 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.att2 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.att3 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.A_adp = A_adp(in_c, 16, seq_len_in=seq_len, seq_len_out=35 * 2)
        # self.M_learn = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_adp = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_skel1 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_skel2 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_skel3 = Parameter(torch.FloatTensor(node_n, node_n))
        self.weight_seq = Parameter(torch.FloatTensor(seq_len, seq_len))
        self.graph = Graph(node_n, seq_len)
        self.A_ske = Parameter(torch.from_numpy(self.graph.A_ske.astype(np.float32)), requires_grad=False)
        self.att = Parameter(torch.from_numpy(self.graph.A_ske.astype(np.float32)))
        nn.init.constant_(self.att, 1e-6)
        # self.reshape_conv = torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1))
        # self.weight_c = Parameter(torch.FloatTensor(in_c, out_c))
        # self.linear=nn.Linear(in_c,out_c,bias=True)
        self.mult_head = nn.ModuleList()
        self.mult_weight = nn.ModuleList()
        self.alpha = Parameter(torch.zeros(1))
        for i in range(3):
            self.mult_head.append(A_adp(in_c, out_c, node_n, seq_len_in=seq_len, seq_len_out=35 * 2))
            self.mult_weight.append(nn.Conv2d(in_c, out_c, 1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.support = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(2))
        # self.weight_c.data.uniform_(-stdv, stdv)
        self.weight_seq.data.uniform_(-stdv, stdv)
        # self.att1.data.uniform_(-stdv, stdv)
        # self.att2.data.uniform_(-stdv, stdv)
        # self.att3.data.uniform_(-stdv, stdv)
        # self.M_skel1.data.uniform_(-stdv, stdv)
        # self.M_skel2.data.uniform_(-stdv, stdv)
        # self.M_skel3.data.uniform_(-stdv, stdv)
        # self.M_adp.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, c, n, l = input.shape
        A_learn = self.A_ske + self.att
        output_gcn = None
        for i in range(3):
            A_adp = self.mult_head[i](input)
            # A_adp = torch.mul(A_adp, self.alpha)
            A = A_learn[i] + A_adp
            # input [b,c,22,35]
            # 先进行图卷积再进行空域卷积
            # [b,c,22,35] -> [b,35,22,c] -> [b,35,22,c]
            # support = torch.matmul(input.permute(0, 3, 2, 1), self.weight_c)
            support = torch.matmul(input.permute(0, 1, 3, 2), A)
            head_output = self.mult_weight[i](support).permute(0, 1, 3, 2)  # b l c n
            output_gcn = output_gcn + head_output if output_gcn is not None else head_output

        # [b,35,22,c] -> [b,35,22,64]

        # 进行空域卷积
        # [b,35,22,64] -> [b,22,64,35]
        output_fc = torch.matmul(output_gcn, self.weight_seq).contiguous()
        # res = self.reshape_conv(input)
        if self.bias is not None:
            return (output_fc + self.bias)
        else:
            return output_fc

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SGraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_c, out_c, node_n=22, seq_len=35, bias=True):
        super(SGraphConvolution, self).__init__()
        self.in_features = in_c
        self.out_features = out_c
        # self.att1 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.att2 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.att3 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.A_adp = A_adp(in_c, 16, seq_len_in=seq_len, seq_len_out=35 * 2)
        # self.M_learn = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_adp = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_skel1 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_skel2 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_skel3 = Parameter(torch.FloatTensor(node_n, node_n))
        self.weight_seq = Parameter(torch.FloatTensor(seq_len, seq_len))
        self.graph = Graph(node_n, seq_len)
        self.A_ske = Parameter(torch.from_numpy(self.graph.A_ske.astype(np.float32)), requires_grad=False)
        self.att = Parameter(torch.from_numpy(self.graph.A_ske.astype(np.float32)))
        nn.init.constant_(self.att, 1e-6)
        # self.reshape_conv = torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1))
        # self.weight_c = Parameter(torch.FloatTensor(in_c, out_c))
        # self.linear=nn.Linear(in_c,out_c,bias=True)
        self.mult_head = nn.ModuleList()
        self.mult_weight = nn.ModuleList()
        self.alpha = Parameter(torch.zeros(1))
        for i in range(3):
            self.mult_head.append(A_adp(in_c, out_c, node_n, seq_len_in=seq_len, seq_len_out=35 * 2))
            self.mult_weight.append(nn.Conv2d(in_c, out_c, 1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.support = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(2))
        # self.weight_c.data.uniform_(-stdv, stdv)
        self.weight_seq.data.uniform_(-stdv, stdv)
        # self.att1.data.uniform_(-stdv, stdv)
        # self.att2.data.uniform_(-stdv, stdv)
        # self.att3.data.uniform_(-stdv, stdv)
        # self.M_skel1.data.uniform_(-stdv, stdv)
        # self.M_skel2.data.uniform_(-stdv, stdv)
        # self.M_skel3.data.uniform_(-stdv, stdv)
        # self.M_adp.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, c, n, l = input.shape
        A_learn = self.A_ske + self.att
        output_gcn = None
        for i in range(3):
            # A_adp = self.mult_head[i](input)
            A_adp = self.mult_head[i](input)
            A = A_learn[i] + A_adp
            # input [b,c,22,35]
            # 先进行图卷积再进行空域卷积
            # [b,c,22,35] -> [b,35,22,c] -> [b,35,22,c]
            # support = torch.matmul(input.permute(0, 3, 2, 1), self.weight_c)
            support = torch.matmul(input.permute(0, 1, 3, 2), A)
            head_output = self.mult_weight[i](support).permute(0, 1, 3, 2)  # b l c n
            output_gcn = output_gcn + head_output if output_gcn is not None else head_output

        # [b,35,22,c] -> [b,35,22,64]

        # 进行空域卷积
        # [b,35,22,64] -> [b,22,64,35]
        # output_fc = torch.matmul(output_gcn, self.weight_seq).contiguous()
        # res = self.reshape_conv(input)
        if self.bias is not None:
            return (output_gcn + self.bias)
        else:
            return output_gcn

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TGraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_c, out_c, node_n=22, seq_len=35, bias=True):
        super(TGraphConvolution, self).__init__()
        self.in_features = in_c
        self.out_features = out_c
        # self.att1 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.att2 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.att3 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.A_adp = A_adp(in_c, 16, seq_len_in=seq_len, seq_len_out=35 * 2)
        # self.M_learn = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_adp = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_skel1 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_skel2 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.M_skel3 = Parameter(torch.FloatTensor(node_n, node_n))
        self.weight_seq = Parameter(torch.FloatTensor(seq_len, seq_len))
        self.graph = Graph(node_n, seq_len)
        self.A_tem = Parameter(torch.from_numpy(self.graph.A_tem.astype(np.float32)), requires_grad=False)
        self.att = Parameter(torch.from_numpy(self.graph.A_tem.astype(np.float32)))
        nn.init.constant_(self.att, 1e-6)
        # self.reshape_conv = torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1))
        # self.weight_c = Parameter(torch.FloatTensor(in_c, out_c))
        # self.linear=nn.Linear(in_c,out_c,bias=True)
        self.mult_head = nn.ModuleList()
        self.mult_weight = nn.ModuleList()
        self.alpha = Parameter(torch.zeros(1))
        for i in range(2):
            self.mult_head.append(TA_adp(in_c, out_c, node_n, seq_len_in=seq_len, seq_len_out=35 * 2))
            self.mult_weight.append(nn.Conv2d(in_c, out_c, 1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(node_n))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.support = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(2))
        # self.weight_c.data.uniform_(-stdv, stdv)
        self.weight_seq.data.uniform_(-stdv, stdv)
        # self.att1.data.uniform_(-stdv, stdv)
        # self.att2.data.uniform_(-stdv, stdv)
        # self.att3.data.uniform_(-stdv, stdv)
        # self.M_skel1.data.uniform_(-stdv, stdv)
        # self.M_skel2.data.uniform_(-stdv, stdv)
        # self.M_skel3.data.uniform_(-stdv, stdv)
        # self.M_adp.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, c, l, n = input.shape
        A_learn = self.A_tem + self.att
        output_gcn = None
        for i in range(2):
            A_adp = self.mult_head[i](input)
            # A_adp = torch.mul(A_adp, self.alpha)
            A = A_learn[i] + A_adp
            # input [b,c,22,35]
            # 先进行图卷积再进行空域卷积
            # [b,c,22,35] -> [b,35,22,c] -> [b,35,22,c]
            # support = torch.matmul(input.permute(0, 3, 2, 1), self.weight_c)
            support = torch.matmul(input.permute(0, 1, 3, 2), A)
            head_output = self.mult_weight[i](support).permute(0, 1, 3, 2)  # b l c n
            output_gcn = output_gcn + head_output if output_gcn is not None else head_output

        # [b,35,22,c] -> [b,35,22,64]

        # 进行空域卷积
        # [b,35,22,64] -> [b,22,64,35]
        # output_fc = torch.matmul(output_gcn, self.weight_seq).contiguous()
        # res = self.reshape_conv(input)
        if self.bias is not None:
            return (output_gcn + self.bias)
        else:
            return output_gcn

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
        self.a = nn.AdaptiveAvgPool1d(10)

    def forward(self, x):
        b, n, l, c = x.shape
        y = self.squeeze(x.reshape(b, n * l, c)).reshape(b, n, l)
        y = self.excitation(y).view(b, n, l, 1)
        return x * y.expand_as(x)


def mish(x):
    return (x * torch.tanh(F.softplus(x)))


class MlpBlock(nn.Module):
    def __init__(self, mlp_hidden_dim, mlp_input_dim, mlp_bn_dim, activation='gelu', regularization=0,
                 initialization='none'):
        super().__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_bn_dim = mlp_bn_dim
        # self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_input_dim)
        # self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim)
        self.gc1 = TGraphConvolution(16, 32, node_n=23, seq_len=16, bias=True)
        self.gc2 = TGraphConvolution(32, 16, node_n=23, seq_len=16, bias=True)
        # self.fc2 = nn.Linear(self.mlp_hidden_dim, self.mlp_input_dim)
        if regularization > 0.0:
            self.reg1 = nn.Dropout(regularization)
            self.reg2 = nn.Dropout(regularization)
        elif regularization == -1.0:
            self.reg1 = nn.BatchNorm1d(self.mlp_bn_dim)
            self.reg2 = nn.BatchNorm1d(self.mlp_bn_dim)
        else:
            self.reg1 = None
            self.reg2 = None

        if activation == 'gelu':
            self.act1 = nn.GELU()
        elif activation == 'mish':
            self.act1 = mish  # nn.Mish()
        else:
            raise ValueError('Unknown activation function type: %s' % activation)

    def forward(self, x):
        x = self.gc1(x.permute(0, 3, 2, 1))
        x = self.act1(x)
        if self.reg1 is not None:
            x = self.reg1(x)
        x = self.gc2(x)
        if self.reg2 is not None:
            x = self.reg2(x)

        return x


class SMlpBlock(nn.Module):
    def __init__(self, mlp_hidden_dim, mlp_input_dim, mlp_bn_dim, activation='gelu', regularization=0,
                 initialization='none'):
        super().__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_bn_dim = mlp_bn_dim
        # self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_input_dim)
        # self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim)
        self.gc1 = SGraphConvolution(16, 16, node_n=23, seq_len=16, bias=True)
        self.gc2 = SGraphConvolution(16, 16, node_n=23, seq_len=16, bias=True)
        # self.fc2 = nn.Linear(self.mlp_hidden_dim, self.mlp_input_dim)
        if regularization > 0.0:
            self.reg1 = nn.Dropout(regularization)
            self.reg2 = nn.Dropout(regularization)
        elif regularization == -1.0:
            self.reg1 = nn.BatchNorm1d(self.mlp_bn_dim)
            self.reg2 = nn.BatchNorm1d(self.mlp_bn_dim)
        else:
            self.reg1 = None
            self.reg2 = None

        if activation == 'gelu':
            self.act1 = nn.GELU()
        elif activation == 'mish':
            self.act1 = mish  # nn.Mish()
        else:
            raise ValueError('Unknown activation function type: %s' % activation)

    def forward(self, x):
        x = self.gc1(x)
        x = self.act1(x)
        if self.reg1 is not None:
            x = self.reg1(x)
        x = self.gc2(x)
        if self.reg2 is not None:
            x = self.reg2(x)

        return x


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=True):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.seq_len, self.hidden_dim,
                                               activation=activation, regularization=regularization,
                                               initialization=initialization)
        self.mlp_block_channel_mixing = SMlpBlock(self.channels_mlp_dim, self.hidden_dim, self.seq_len,
                                                  activation=activation, regularization=regularization,
                                                  initialization=initialization)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)

        self.LN1 = nn.LayerNorm(16)
        self.LN2 = nn.LayerNorm(16)

    def forward(self, x):
        # shape x [256, 8, 512] [bs, patches/time_steps, channels
        y = self.LN1(x.permute(0, 2, 3, 1))

        # y = y.transpose(1, 2)
        y = self.mlp_block_token_mixing(y)
        # y = y.transpose(1, 2)
        y = y.permute(0, 3, 2, 1)
        if self.use_se:
            y = self.se(y)
        x = x.permute(0, 2, 3, 1) + y

        y = self.LN2(x).permute(0, 3, 1, 2)
        y = self.mlp_block_channel_mixing(y)

        if self.use_se:
            y = self.se(y.permute(0, 2, 3, 1))
        y = x + y

        return y.permute(0, 3, 1, 2)


class MixerBlock_Channel(nn.Module):
    def __init__(self, channels_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=True):
        super().__init__()
        self.channels_mlp_dim = channels_mlp_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_channel_mixing = MlpBlock(self.channels_mlp_dim, self.hidden_dim, self.seq_len,
                                                 activation=activation, regularization=regularization,
                                                 initialization=initialization)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)

        self.LN2 = nn.LayerNorm(self.hidden_dim)

        # self.act1 = nn.GELU()

    def forward(self, x):
        # shape x [256, 8, 512] [bs, patches/time_steps, channels]
        y = x

        if self.use_se:
            y = self.se(y)
        x = x + y
        y = self.LN2(x)
        y = self.mlp_block_channel_mixing(y)
        if self.use_se:
            y = self.se(y)

        return x + y


class MixerBlock_Token(nn.Module):
    def __init__(self, tokens_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=True):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.seq_len, self.hidden_dim,
                                               activation=activation, regularization=regularization,
                                               initialization=initialization)

        self.use_se = use_se

        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)

        self.LN1 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        # shape x [256, 8, 512] [bs, patches/time_steps, channels]
        y = self.LN1(x)
        y = y.transpose(1, 2)
        y = self.mlp_block_token_mixing(y)
        y = y.transpose(1, 2)

        if self.use_se:
            y = self.se(y)
        x = x + y

        return x + y


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, hidden_dim, tokens_mlp_dim,
                 channels_mlp_dim, seq_len, pred_len, activation='gelu',
                 mlp_block_type='normal', regularization=0, input_size=51,
                 initialization='none', r_se=4, use_max_pooling=False,
                 use_se=False):

        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.input_size = input_size  # varyies with the number of joints
        self.sgcn = SGraphConvolution(3, 16, node_n=23, seq_len=16, bias=True)
        self.activation = activation

        self.channel_only = False  # False #True
        self.token_only = False  # False #True

        if self.channel_only:
            self.Mixer_Block = nn.ModuleList(MixerBlock_Channel(self.channels_mlp_dim, self.seq_len, self.hidden_dim,
                                                                activation=self.activation,
                                                                regularization=regularization,
                                                                initialization=initialization,
                                                                r_se=r_se, use_max_pooling=use_max_pooling,
                                                                use_se=use_se)
                                             for _ in range(num_blocks))

        if self.token_only:

            self.Mixer_Block = nn.ModuleList(MixerBlock_Token(self.tokens_mlp_dim, self.seq_len, self.hidden_dim,
                                                              activation=self.activation, regularization=regularization,
                                                              initialization=initialization,
                                                              r_se=r_se, use_max_pooling=use_max_pooling, use_se=use_se)
                                             for _ in range(num_blocks))

        else:

            self.Mixer_Block = nn.ModuleList(MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim,
                                                        self.seq_len, self.hidden_dim, activation=self.activation,
                                                        regularization=regularization, initialization=initialization,
                                                        r_se=r_se, use_max_pooling=use_max_pooling, use_se=use_se)
                                             for _ in range(num_blocks))

        self.LN = nn.LayerNorm(16)

        # self.sgcn_out = SGraphConvolution(16, 3, node_n=23, seq_len=14, bias=True)

        self.pred_len = pred_len
        self.conv_out = nn.Linear(self.seq_len, self.pred_len)
        self.sgcn_out = SGraphConvolution(16, 3, node_n=23, seq_len=self.pred_len, bias=True)
        self.tgcn_out = TGraphConvolution(3, 3, node_n=23, seq_len=self.pred_len, bias=True)
        # a=1

    def forward(self, x):
        # x = x.unsqueeze(1)
        y = self.sgcn(x)
        # y = y.squeeze(dim=3).transpose(1, 2)

        # [256, 8, 512] [bs, patches/time_steps, channels]
        for mb in self.Mixer_Block:
            y = mb(y)
        y = self.LN(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.conv_out(y)
        out = self.sgcn_out(y)
        # out = out.permute(0, 1, 3, 2)
        # out = self.tgcn_out(out)
        # out = out.permute(0, 1, 3, 2)
        # a=1

        return out
