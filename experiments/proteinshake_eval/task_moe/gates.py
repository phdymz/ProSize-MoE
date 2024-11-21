import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear



def gumbel_rsample(shape, device: torch.device):
    one = torch.tensor(1.0, device=device)
    zero = torch.tensor(0.0, device=device)
    gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
    return gumbel(shape)


def normal_rsample(shape, device, num_expert):

    std = torch.tensor(1.0/num_expert, device=device)
    mean = torch.tensor(0.0, device=device)
    normal = torch.distributions.normal.Normal(mean, std).rsample  # type: ignore

    return normal(shape)


def one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros([data.size(0), num_classes],
                         device=data.device,
                         dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), 1)
    return result


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample
    return x * uniform(x.shape)



class TopKGate(nn.Module):
    def __init__(self, model_dim, num_experts = 4, k = 2, noisy_gate_policy = 'Jitter', cfg = None, moe_type = None):
        super().__init__( )

        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.model_dim = model_dim
        self.k = k
        self.cfg = cfg
        self.noisy_gate_policy = noisy_gate_policy
        self.moe_type = moe_type
        self.prompt_dim = 64
        self.wg = Linear(model_dim + self.prompt_dim, num_experts, bias=True).float()


    def forward(self, input, prompt=None):
        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(torch.cat([input_fp32, prompt], dim=-1))

        gate_output = self.top2gating(
                logits,
                self.noisy_gate_policy if self.training else None)

        return gate_output


    def top2gating(self, logits, noisy_gate_policy = None):
        """Implements Top2Gating on logits."""
        # everything is in fp32 in this function

        b, num_experts = logits.shape

        logits_w_noise = None
        if noisy_gate_policy == 'RSample':
            logits_w_noise = logits + gumbel_rsample(logits.shape,
                                                     device=logits.device) * self.noise_std
        elif noisy_gate_policy == 'vmoe':
            logits_w_noise = logits + normal_rsample(logits.shape,
                                                     device=logits.device,
                                                     num_expert=num_experts/self.noise_std)

        # topk_indices = torch.topk(logits, self.k, dim=1).indices
        topk_indices = torch.topk(
            logits_w_noise
            if logits_w_noise is not None else logits,
            self.k,
            dim=-1).indices

        indices_s = [x for x in topk_indices.chunk(self.k, dim=-1)]
        masks_se = [
            one_hot_with_dtype(x.view(-1), num_classes=num_experts, dtype=x.dtype).view(b, num_experts)
            for x in indices_s
        ]

        if noisy_gate_policy == 'vmoe':
            gates = F.softmax(logits_w_noise, dim=1)

        else:
            gates = F.softmax(logits, dim=-1)

        # self.load_balance(gates, masks_se[0], num_experts)
        gates_s = [(gates * x).sum(dim=-1) for x in masks_se]


        if self.k > 1:
            # Normalize Gate
            denom_s = torch.clamp(sum(gates_s),
                                  min=torch.finfo(gates_s[0].dtype).eps)
            gates_s = [x / denom_s for x in gates_s]

        # self.tb_output(mask1=None, exp_counts=None, gates=gates_s)

        return masks_se, gates_s

