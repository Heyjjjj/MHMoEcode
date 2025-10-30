import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np
from einops import rearrange

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
        init.normal_(self.fc1.weight, std=0.001)
        init.constant_(self.fc1.bias, 0)
        init.normal_(self.fc2.weight, std=0.001)
        init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

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
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

        

class Adapter_MoElayer(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    ['cv', 'cd', 'ad', 'rd', 'scd']
    """

    def __init__(self, dim, num_experts=5, noisy_gating=True, k=1, num_layers=1, mlp_ratio=4,expert_layers=None,expert_type='conv'):
        super(Adapter_MoElayer, self).__init__()
        self.noisy_gating = noisy_gating
        self.dim = dim
        self.k = k
        self.identity = nn.Identity()
        self.expert_type = expert_type
        
        if expert_layers is not None:
            self.adapter_experts = expert_layers
            self.num_experts = len(expert_layers)
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.adapter_experts = nn.ModuleList([
                ExpertMLP(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
                for _ in range(num_experts)])
            self.num_experts = num_experts

        # define adapter param
        self.num_experts = len(self.adapter_experts)
        self.w_gate = nn.Parameter(torch.zeros(dim,  self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(dim,  self.num_experts), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        assert(self.k <= self.num_experts)
    
    def count_parameters(self):
        """计算MoE层的参数数量"""
        total_params = 0
        
        # 计算门控网络的参数
        gate_params = sum(p.numel() for p in [self.w_gate, self.w_noise])
        total_params += gate_params
        
        # 计算专家的参数
        expert_params = 0
        for expert in self.adapter_experts:
            expert_params += sum(p.numel() for p in expert.parameters())
        
        # MoE总参数 = 门控参数 + 专家参数
        total_params += expert_params
        
        return total_params, gate_params, expert_params, len(self.adapter_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Input to noisy_top_k_gating contains NaN or Inf")
            
        clean_logits = x @ self.w_gate
        
        if torch.isnan(clean_logits).any() or torch.isinf(clean_logits).any():
            print("Warning: clean_logits contains NaN or Inf")
            
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
            
        return gates, load

    def forward(self, x, loss_coef=1):
        """Args:
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        b, h, w, c = x.shape
        #print('x',x.dtype)
        x_global = torch.mean(x, dim=[1, 2])
        #print('x',x)
        #print('x_global',x_global)

        gates, load = self.noisy_top_k_gating(x_global, self.training)

        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = []
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0: continue
            if self.expert_type == 'conv':
                expert_input = rearrange(expert_inputs[i], '(n) h w c -> n c h w')
                expert_output = self.adapter_experts[i](expert_input)
                expert_output = rearrange(expert_output, 'n c h w -> n (c h w)')
            else:
                expert_output = self.adapter_experts[i](expert_inputs[i])
            expert_outputs.append(expert_output)

        y = dispatcher.combine(expert_outputs)
        y = rearrange(y, 'b (c h w) -> b h w c', h=h, w=w, c=c)

        return y, loss

class MLP_MoElayer(nn.Module):
    """Sparsely gated mixture of experts layer with MLP networks as experts.
    Args:
    dim: integer - size of the input and output
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    mlp_ratio: ratio for hidden size to input size
    """

    def __init__(self, dim, num_experts=5, noisy_gating=True, k=1, mlp_ratio=4):
        super(MLP_MoElayer, self).__init__()
        self.noisy_gating = noisy_gating
        self.dim = dim
        self.k = k
        self.num_experts = num_experts
        self.mlp_ratio = mlp_ratio
        
        # Create MLP experts
        hidden_size = int(dim * mlp_ratio)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, dim),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])
        
        # Gate parameters
        self.w_gate = nn.Parameter(torch.zeros(dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(dim, num_experts), requires_grad=True)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.w_gate)
        nn.init.xavier_uniform_(self.w_noise)
        
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        
        assert self.k <= self.num_experts
    
    def count_parameters(self):
        """计算MoE层的参数数量"""
        total_params = 0
        
        # 计算门控网络的参数
        gate_params = sum(p.numel() for p in [self.w_gate, self.w_noise])
        total_params += gate_params
        
        # 计算专家的参数
        expert_params = 0
        for expert in self.experts:
            expert_params += sum(p.numel() for p in expert.parameters())
        
        # MoE总参数 = 门控参数 + 专家参数
        total_params += expert_params
        
        return total_params, gate_params, expert_params, len(self.experts)
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] <= 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        
        x_double = x.double()
        mean = x_double.mean()
        var = x_double.var()
        
        if abs(mean) < eps:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        
        return (var / (mean**2 + eps)).to(x.dtype)
    
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
    
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
        x: input Tensor with shape [batch_size, input_size]
        train: a boolean - we only add noise at training time.
        noise_epsilon: a float
        Returns:
        gates: a Tensor with shape [batch_size, num_experts]
        load: a Tensor with shape [num_experts]
        """

        
        clean_logits = x @ self.w_gate
        

        
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise_stddev = torch.clamp(noise_stddev, min=1e-6, max=10.0)
            
            noise = torch.randn_like(clean_logits) * noise_stddev
            noise = torch.clamp(noise, -10.0, 10.0)
            
            noisy_logits = clean_logits + noise
            logits = noisy_logits
        else:
            logits = clean_logits
        
        logits = torch.clamp(logits, -50.0, 50.0)
        
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        
        top_k_gates = F.softmax(top_k_logits, dim=1)
        
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        
        
        return gates, load
    
    def forward(self, x, loss_coef=1):
        """Args:
        x: tensor shape [batch_size, height, width, channels]
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, height, width, channels]
        loss: a scalar. This should be added into the overall training loss of the model.
        """
        B, H, W, C  = x.shape
 
        

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        

        x_flat = x.reshape(B, H*W, C)

        x_global = torch.mean(x, dim=[1, 2])  # [B, C]
        
        if torch.isnan(x_global).any() or torch.isinf(x_global).any():
            print("Warning: x_global contains NaN or Inf")
            x_global = torch.nan_to_num(x_global, nan=0.0, posinf=1.0, neginf=0.0)
        
        gates, load = self.noisy_top_k_gating(x_global, self.training)
        

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss = torch.clamp(loss, 0.0, 1000.0)
        loss *= loss_coef

        x_flat = x_flat.reshape(B * H * W, C)
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            mask = (gates[:, expert_idx] > 0)
            
            if mask.any():
                mask_expanded = mask.unsqueeze(1).expand(-1, H*W).reshape(-1)
                expert_input = x_flat[mask_expanded]
                expert_output = self.experts[expert_idx](expert_input)
                
                if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
                    print(f"Warning: expert_output for expert {expert_idx} contains NaN or Inf")
                    expert_output = torch.nan_to_num(expert_output, nan=0.0, posinf=1.0, neginf=0.0)

                gate_weights = gates[mask, expert_idx]
                gate_weights_expanded = gate_weights.unsqueeze(1).expand(-1, H*W).reshape(-1, 1)
                weighted_output = expert_output * gate_weights_expanded
                output[mask_expanded] += weighted_output
        
        output = output.reshape(B, H, W, C)

        
        return output, loss