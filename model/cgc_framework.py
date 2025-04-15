import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import *


class ExpertModule(nn.Module):
    def __init__(self, experts_hidden, expert_type=None, dropout_rate=0.1, num_experts=5, num_tasks=2):
        super(ExpertModule, self).__init__()

        self.num_experts = num_experts
        self.expert_type = expert_type

        if self.expert_type == "protein":
            self.share_expert = nn.ModuleList(
                [ProteinExpert(hid_dim=experts_hidden) for _ in range(num_experts)])
            self.task_experts = nn.ModuleList([])
            for _ in range(num_tasks):
                task_expert = nn.ModuleList(
                    [ProteinExpert(hid_dim=experts_hidden) for _ in range(num_experts)])
                self.task_experts.append(task_expert)

        elif self.expert_type == "reaction":
            self.share_expert = nn.ModuleList(
                [ReactionExpert() for _ in range(num_experts)])
            self.task_experts = nn.ModuleList([])
            for _ in range(num_tasks):
                task_expert = nn.ModuleList(
                    [ReactionExpert() for _ in range(num_experts)])
                self.task_experts.append(task_expert)
        else:
            raise ValueError(f"Unsupported expert type: {expert_type}")


    def forward(self, share_x, task_x):
        assert len(task_x) == len(self.task_experts)
        # 蛋白(b,1,128),反应(b,2,512)

        share_expert_out = [e(share_x) for e in self.share_expert]
        if self.expert_type == "protein":
            share_expert_out = torch.concat(share_expert_out, dim=0).view(-1, self.num_experts, 1, 128)
        elif self.expert_type == "reaction":
            share_expert_out = torch.concat(share_expert_out, dim=0).view(-1, self.num_experts, 2, 512)

        task_expert_out_list = []
        for i, task_expert in enumerate(self.task_experts): # 2个
            if task_x[i] is not None:
                task_expert_out = [e(task_x[i]) for e in task_expert]
                if self.expert_type == "protein":
                    task_expert_out = torch.concat(
                        task_expert_out, dim=0).view(-1, self.num_experts, 1, 128)
                elif self.expert_type == "reaction":
                    task_expert_out = torch.concat(
                        task_expert_out, dim=0).view(-1, self.num_experts, 2, 512)
            else:
                task_expert_out = None
            task_expert_out_list.append(task_expert_out)

        return share_expert_out, task_expert_out_list
class Gate(nn.Module):
    def __init__(self, input_size, output_size):
        super(Gate, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):

        return self.fc1(x)

class GateModule(nn.Module):
    def __init__(self, num_experts, num_tasks, gate_type=None):
        super(GateModule, self).__init__()
        self.gate_type = gate_type
        if self.gate_type == "protein":
            gate_in = 128
        elif self.gate_type == "reaction":
            gate_in = 512

        self.share_gate = Gate(gate_in, num_experts * 2)
        self.task_gates = nn.ModuleList(
            [Gate(gate_in, num_experts * 2) for _ in range(num_tasks)]
        )
        self.gate_activation = nn.Softmax(dim=-1)

    def forward(self, share_x, task_x):
        assert len(task_x) == len(self.task_gates)
        
        share_x = share_x.squeeze(1)
        if self.gate_type == "reaction":
            share_x = torch.mean(share_x, dim=1)
   
        share_gate_out = self.share_gate(share_x)
        share_gate_out = self.gate_activation(share_gate_out)

        task_gate_out_list = []   
        for i, e in enumerate(self.task_gates):
            if task_x[i] is not None:
                task_x[i] = task_x[i].squeeze(1)
                if self.gate_type == "reaction":
                    task_x[i] = torch.mean(task_x[i], dim=1)
                task_gate_out = e(task_x[i])
            else: 
                task_gate_out = None

            task_gate_out_list.append(task_gate_out)

        return share_gate_out, task_gate_out_list


class PleLayer(nn.Module):

    def __init__(
            self, experts_hidden, dropout_rate=0.1,
            num_experts=None, num_tasks=2, plelayer_type=None
    ):
        super(PleLayer, self).__init__()
        plelayer_type = plelayer_type
        self.experts = ExpertModule(
            experts_hidden, expert_type=plelayer_type, dropout_rate=dropout_rate,
              num_experts=num_experts, num_tasks=num_tasks)
    
        self.gates = GateModule(num_experts, num_tasks, gate_type=plelayer_type)
        self._changed_shape = False

    def _change_shape(self, share_x, task_x):
        if share_x.dim() == 4:
            if share_x.size(-1) == 128:
                share_x = share_x.view(-1, 1, 128)
                for i, t in enumerate(task_x):
                    if t is not None:
                        task_x[i] = t.view(-1, 1, 128)

            elif share_x.size(-1) == 512:
                share_x = share_x.view(-1, 2, 512)
                for i, t in enumerate(task_x):
                    if t is not None:
                        task_x[i] = t.view(-1, 2, 512)

            self._changed_shape = True
        return share_x, task_x
    
    def _reshape(self, share_out, task_out_list, original_shape):
        if share_out.dim() == 3:
            share_out = share_out.view(original_shape)
            for i, t in enumerate(task_out_list):
                if t is not None:
                    task_out_list[i] = t.view(original_shape)
            self._changed_shape = False
        return share_out, task_out_list

    def forward(self, share_x, task_x):
        if share_x.shape[1] > 10:
            original_shape = share_x.shape
            share_x, task_x = self._change_shape(share_x, task_x)
            
        share_expert_out, task_expert_out_list = self.experts(share_x, task_x)
        share_gate_out, task_gate_out_list = self.gates(share_x, task_x)

        task_out_list = []
        for i in range(len(task_x)):
            if task_expert_out_list[i] is not None:
                task_expert_out = task_expert_out_list[i]
                task_gate_out = task_gate_out_list[i]

                task_out = torch.cat([share_expert_out, task_expert_out], dim=1)
                task_out = torch.einsum('be,beuv -> beuv', task_gate_out, task_out)
                task_out = task_out.sum(dim=1)
            else:
                task_out = None
            task_out_list.append(task_out)
        task_expert_out = [x for x in task_expert_out_list if x is not None]
        share_out = torch.cat([share_expert_out, *task_expert_out], dim=1)
        share_out = torch.einsum('be,beuv -> beuv', share_gate_out, share_out)
        share_out = share_out.sum(dim=1)
        if self._changed_shape:
            share_out, task_out_list = self._reshape(share_out, task_out_list, original_shape)

        return share_out, task_out_list


class PLE_Protein(nn.Module):  # Pipelined Expert Layers
    def __init__(self, experts_hidden=128, dropout_rate=0.1,
                  num_experts=1, num_tasks=2, num_ple_layers=1):
        super(PLE_Protein, self).__init__()

        self.layers = nn.ModuleList([])
        self.num_tasks = num_tasks

        layer = PleLayer(experts_hidden, dropout_rate=dropout_rate, 
                         num_experts=num_experts, num_tasks=num_tasks,plelayer_type="protein")
        self.layers.append(layer)

    def forward(self, ec_x=None, rpre_x=None):
        if ec_x is not None:
            share_x = ec_x.clone()
        elif rpre_x is not None:
            share_x = rpre_x.clone()
        task_x = [ec_x, rpre_x]  # 0:ec_x,1:rpre_x

        for layer in self.layers:
            share_x, task_x = layer(share_x, task_x)

        return task_x

class PLE_Reaction(nn.Module):  # Pipelined Expert Layers
    def __init__(self, experts_hidden=128, dropout_rate=0.1,
                  num_experts=1, num_tasks=2, num_ple_layers=1):
        super(PLE_Reaction, self).__init__()

        self.layers = nn.ModuleList([])
        self.num_tasks = num_tasks

        layer = PleLayer(experts_hidden, dropout_rate=dropout_rate, 
                         num_experts=num_experts, num_tasks=num_tasks,plelayer_type="reaction")
        self.layers.append(layer)

    def forward(self, cls_x=None, rpre_x=None):
        if cls_x is not None:
            share_x = cls_x.clone()
        elif rpre_x is not None:
            share_x = rpre_x.clone()
        task_x = [cls_x, rpre_x]  # 0:cls_x,1:rpre_x

        for layer in self.layers:
            share_x, task_x = layer(share_x, task_x)

        return task_x