import torch
from torch import nn

from .gates import TopKGate

from .experts import FusedExperts as Experts
from torch_geometric.nn.dense.linear import Linear


class TaskMoE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=4,
                 k=2,
                 cfg=None):
        super().__init__()
        self.num_experts = num_experts

        if isinstance(expert, Linear) or isinstance(expert, nn.Linear):
            self.expert_type = 'linear'
        elif isinstance(expert, nn.MultiheadAttention):
            self.expert_type = 'attention'
        else:
            raise NotImplementedError('please check expert type')

        experts = Experts(expert, cfg, num_experts)

        self.gate = TopKGate(hidden_size,
                             num_experts,
                             k,
                             moe_type=self.expert_type)

        self.experts = experts



    def forward(self, hidden_states, mode=None, prompt=None, gate_decision=None, return_gate = False):
        if gate_decision is not None:
            top_indices, gates = gate_decision
        else:
            top_indices, gates = self.gate(hidden_states, prompt)

        expert_output = self.experts(hidden_states, top_indices, gates, mode)

        if return_gate:
            return expert_output, [top_indices, gates]
        else:
            return expert_output


