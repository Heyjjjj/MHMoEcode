import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np
import math
from einops.layers.torch import Rearrange
from einops import rearrange
import math
from inspect import isfunction


class ExpertMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def init_weights(self):
        # 初始化 fc1 的权重和偏置
        init.normal_(self.fc1.weight, std=0.001)
        init.constant_(self.fc1.bias, 0)
        # 初始化 fc2 的权重和偏置
        init.normal_(self.fc2.weight, std=0.001)
        init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)

        x = self.drop(x)

        x = self.fc2(x)

        x = self.drop(x)

        return x

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
    
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=x.device, dtype=conv_weight.dtype)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        out_diff = nn.functional.conv2d(input=x, weight=conv_weight_cd, bias=self.conv.bias,
                                        stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
        return out_diff


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad)
        out_diff = nn.functional.conv2d(input=x, weight=conv_weight_ad, bias=self.conv.bias,
                                        stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
        return out_diff



class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5, device=x.device, dtype=conv_weight.dtype)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias,
                                            stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff



class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=x.device, dtype=conv_weight.dtype)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd)

        out_diff = nn.functional.conv2d(input=x, weight=conv_weight_hd, bias=self.conv.bias,
                                        stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
        return out_diff



class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=x.device, dtype=conv_weight.dtype)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        out_diff = nn.functional.conv2d(input=x, weight=conv_weight_vd, bias=self.conv.bias,
                                        stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
        return out_diff


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=False):
        """Sum together the expert output, weighted by the gates."""
        expert_out_clean = []
        for out in expert_out:
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("Warning: expert_out contains NaN or Inf")
                out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
            expert_out_clean.append(out)
        
        stitched = torch.cat(expert_out_clean, 0)
        
        # 应用 softmax 进行数值稳定
        stitched_softmax = torch.softmax(stitched, dim=1)
        
        if multiply_by_gates:
            stitched_softmax = stitched_softmax.mul(self._nonzero_gates)
        
        zeros = torch.zeros(self._gates.size(0), expert_out_clean[-1].size(1), 
                           requires_grad=True, device=stitched_softmax.device)
        
        # 组合样本
        combined = zeros.index_add(0, self._batch_index, stitched_softmax.float())
        
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
        
class De_MoElayer(nn.Module):
    def __init__(self, dim, num_experts=5, top_k=1, mlp_ratio=4):
        super(De_MoElayer, self).__init__()
        self.router = NoisyTopkRouter(dim, num_experts, top_k)
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_experts = num_experts
        self.expert1 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.expert2 = Conv2d_cd(dim, dim, 3, bias=True)
        self.expert3 = Conv2d_hd(dim, dim, 3, bias=True)
        self.expert4 = Conv2d_vd(dim, dim, 3, bias=True)
        self.expert5 = Conv2d_ad(dim, dim, 3, bias=True)
        self.experts = nn.ModuleList([self.expert1, self.expert2, self.expert3, self.expert4, self.expert5])

        self.top_k = top_k

    def forward(self, x):
        b, h, w, c = x.shape
        x_global = torch.mean(x, dim=[1, 2])
        gating_output, indices = self.router(x_global)
        #print(gating_output.size(), indices.size())
        dispatcher = SparseDispatcher(self.num_experts, gating_output)
        expert_inputs = dispatcher.dispatch(x)

        gates = dispatcher.expert_to_gates()
        expert_outputs = []
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0: continue
            # print(expert_inputs[i].size())
            expert_input = rearrange(expert_inputs[i], 'n h w c -> n c h w')
            expert_output = self.experts[i](expert_input)
            # print(expert_output.size())
            expert_output = rearrange(expert_output, 'n c h w -> n (c h w)')
            expert_outputs.append(expert_output)
        
        y = dispatcher.combine(expert_outputs)
        y = rearrange(y, 'b (c h w) -> b h w c', h=h, w=w, c=c)
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: x_global contains NaN or Inf")
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

        return y
    
    def count_parameters(self):
        """计算MoE层的参数数量"""
        total_params = 0
        
        # 计算路由器的参数
        router_params = sum(p.numel() for p in self.router.parameters())
        total_params += router_params
        
        # 计算专家的参数（只计算一次，因为专家是共享的）
        expert_params = 0
        for expert in self.experts:
            expert_params += sum(p.numel() for p in expert.parameters())
        
        # MoE总参数 = 路由器参数 + 专家参数
        total_params += expert_params
        
        return total_params, router_params, expert_params, len(self.experts)

if __name__ == "__main__":
    x = torch.rand([2,3,224,224]).cuda()
    m = SparseMoE(dim=3).cuda()
    y = m(x)
    print(y.shape)