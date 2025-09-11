import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np


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


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output, train):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)
        
        if train:
        # Noise logits
            noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class SparseMoE(nn.Module):
    def __init__(self, dim, out_dim, num_experts=5, top_k=1, mlp_ratio=4):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(dim, num_experts, top_k)
        self.dim = dim
        self.out_dim = out_dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.experts = nn.ModuleList([ExpertMLP(in_features=self.dim, hidden_features=mlp_hidden_dim, out_features=self.out_dim)
                                      for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        b,h,w,c = x.shape
        gating_output, indices = self.router(x, self.training)

        final_output = torch.zeros(b,h,w,self.out_dim).cuda()
        flat_x = x.reshape(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_x = flat_x[flat_mask]
                expert_output = expert(expert_x)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output