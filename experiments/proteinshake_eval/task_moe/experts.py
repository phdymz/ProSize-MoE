import torch
import copy
import torch.nn.functional as F



class FusedExperts(torch.nn.Module):
    def __init__(self, expert, cfg, num_local_experts=2):
        super(FusedExperts, self).__init__()
        self.cfg = cfg

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        self.bias_merge = self.deepspeed_experts[0].bias is not None

    def mergelayer(self, x, index1, index2, gate1, gate2, mode=None):
        if mode == 'q':
            # hidden_states
            _start = 0
            _end = self.deepspeed_experts[index1].weight.shape[0] // 3
            return F.linear(
                x,
                self.deepspeed_experts[index1].weight[_start:_end, :] * gate1 +
                self.deepspeed_experts[index2].weight[_start:_end, :] * gate2,
                bias=self.deepspeed_experts[index1].bias[_start:_end] * gate1 +
                     self.deepspeed_experts[index2].bias[_start:_end] * gate2
                if self.bias_merge else None,
            )

        elif mode == 'kv':
            # history_states
            _start = self.deepspeed_experts[index1].weight.shape[0] // 3

            return F.linear(
                x,
                self.deepspeed_experts[index1].weight[_start:, :] * gate1 +
                self.deepspeed_experts[index2].weight[_start:, :] * gate2,
                bias=self.deepspeed_experts[index1].bias[_start:] * gate1 +
                     self.deepspeed_experts[index2].bias[_start:] * gate2
                if self.bias_merge else None,
            )

        else:
            return F.linear(
                x,
                self.deepspeed_experts[index1].weight * gate1 +
                self.deepspeed_experts[index2].weight * gate2,
                bias=self.deepspeed_experts[index1].bias * gate1 +
                     self.deepspeed_experts[index2].bias * gate2 if self.bias_merge else None,
            )


    def top2_expert_forward(self, x, masks_se, gates, mode=None):

        # b, l, d = x.shape
        x = torch.cat([layer(x).unsqueeze(-2) for layer in self.deepspeed_experts], dim=-2)
        x = (masks_se[0].unsqueeze(-1) * x).sum(-2) * gates[0].unsqueeze(-1) + (masks_se[1].unsqueeze(-1) * x).sum(-2) * gates[1].unsqueeze(-1)

        # if indices[0].size(0) == 1:
        #     x = self.mergelayer(x.view(-1, d), indices[0].view(-1), indices[1].view(-1), gates[0].view(-1), gates[1].view(-1), mode=mode)
        # else:
        #     raise NotImplementedError('only support one or two modality')
        # x = gates[0] * self.deepspeed_experts[indices[0]](x) + gates[1] * self.deepspeed_experts[indices[1]](x)

        return x

    def forward(self, hidden_states, top_indices=None, gates=None, mode=None):
        # top2
        if len(top_indices) == 2:
            out = self.top2_expert_forward(hidden_states, top_indices, gates, mode)
        else:
            raise NotImplementedError("only support top2 ")

        # assert out.shape[1] == hidden_states.shape[1]

        return out
