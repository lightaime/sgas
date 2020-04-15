import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPS, BasicConv
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np
from gcn.gcn_lib.dense import DilatedKnn2d


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)

    def forward(self, x, edge_index, weights, selected_idx=None):
        if selected_idx is None:
            return sum(w * op(x, edge_index) for w, op in zip(weights, self._ops))
        else: # unchosen operations are pruned
            return self._ops[selected_idx](x, edge_index)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, k=9, d=1):
        super(Cell, self).__init__()
        self.preprocess0 = BasicConv([C_prev_prev, C], 'relu', 'batch', bias=False)
        self.preprocess1 = BasicConv([C_prev, C], 'relu', 'batch', bias=False)
        self._steps = steps
        self._multiplier = multiplier
        self.dilated_knn_graph = DilatedKnn2d(k=k, dilation=d)
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, selected_idxs=None):
        edge_index = self.dilated_knn_graph(s0)
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            o_list = []
            for j, h in enumerate(states):
                if selected_idxs[offset + j] == -1: # undecided mix edges
                    o = self._ops[offset + j](h, edge_index, weights[offset + j])   # call the gcn module,
                elif selected_idxs[offset + j] == PRIMITIVES.index('none'): # pruned edges
                    pass
                else: # decided discrete edges
                    o = self._ops[offset + j](h, edge_index, None, selected_idxs[offset + j])
                o_list.append(o)
            s = sum(o_list)
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, num_cells, criterion,
                 steps=4, multiplier=4, stem_multiplier=3,
                 in_channels=3, emb_dims=1024, dropout=0.5, k=9):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = num_cells
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self._in_channels = in_channels
        self._emb_dims = emb_dims
        self._dropout = dropout
        self._k = k
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            BasicConv([in_channels, C_curr], None, 'batch', bias=False),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        for i in range(self._layers):
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, k=k, d=i+1)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.fusion_conv = BasicConv([stem_multiplier*C + C*multiplier*self._layers, emb_dims],
                                     act='leakyrelu', norm='batch', bias=False)
        self.classifier = nn.Sequential(BasicConv([emb_dims*2, 512], act='leakyrelu', norm='batch'),
                                        torch.nn.Dropout(p=dropout),
                                        BasicConv([512, 256], act='leakyrelu', norm='batch'),
                                        torch.nn.Dropout(p=dropout),
                                        BasicConv([256, num_classes], act=None, norm=None))

        self._initialize_alphas()

        self.normal_selected_idxs = torch.tensor(len(self.alphas_normal) * [-1], requires_grad=False, dtype=torch.int)
        self.normal_candidate_flags = torch.tensor(len(self.alphas_normal) * [True], 
                                                   requires_grad=False, dtype=torch.bool)

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self._steps,
                            in_channels=self._in_channels,
                            emb_dims=self._emb_dims, dropout=self._dropout, k=self.k).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        model_new.normal_selected_idxs = self.normal_selected_idxs
        return model_new

    def forward(self, x):
        s0 = s1 = self.stem(x)
        pre_layers = [s1]
        for i, cell in enumerate(self.cells):
            weights = []
            n = 2
            start = 0
            for _ in range(self._steps):
                end = start + n
                for j in range(start, end):
                    weights.append(F.softmax(self.alphas_normal[j], dim=-1))
                start = end
                n += 1

            selected_idxs = self.normal_selected_idxs
            s0, s1 = s1, cell(s0, s1, weights, selected_idxs)
            pre_layers.append(s1)

        fusion = torch.cat(pre_layers, dim=1)
        fusion = self.fusion_conv(fusion)
        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        logits = self.classifier(torch.cat((x1, x2), dim=1))
        return logits.squeeze(-1).squeeze(-1)

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)
        self.alphas_normal = []
        for i in range(self._steps):
            for n in range(2 + i):
                self.alphas_normal.append(Variable(1e-3 * torch.randn(num_ops).cuda(), requires_grad=True))
        self._arch_parameters = [
            self.alphas_normal,
        ]

    def arch_parameters(self):
        return self.alphas_normal

    def check_edges(self, flags, selected_idxs):
        n = 2
        max_num_edges = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            if num_selected_edges >= max_num_edges:
                for j in range(start, end):
                    if flags[j]:
                        flags[j] = False
                        selected_idxs[j] = PRIMITIVES.index('none') # pruned edges
                        self.alphas_normal[j].requires_grad = False
                    else:
                        pass
            start = end
            n += 1

        return flags, selected_idxs

    def parse_gene(self, selected_idxs):
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            for j in range(start, end):
                if selected_idxs[j] == 0:
                    pass
                elif selected_idxs[j] == -1:
                    raise Exception("Contain undecided edges")
                else:
                    gene.append((PRIMITIVES[selected_idxs[j]], j - start))
            start = end
            n += 1

        return gene

    def parse_gene_force(self, flags, selected_idxs, alphas):
        gene = []
        n = 2
        max_num_edges = 2
        start = 0
        mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
        importance = torch.sum(mat[:, 1:], dim=-1)
        masked_importance = torch.min(importance, (2 * flags.float() - 1) * np.inf)
        for _ in range(self._steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            num_edges_to_select = max_num_edges - num_selected_edges
            if num_edges_to_select > 0:
                post_select_edges = torch.topk(masked_importance[start: end], k=num_edges_to_select).indices + start
            else:
                post_select_edges = []
            for j in range(start, end):
                if selected_idxs[j] == 0:
                    pass
                elif selected_idxs[j] == -1:
                    if num_edges_to_select <= 0:
                        raise Exception("Unknown errors")
                    else:
                        if j in post_select_edges:
                            idx = torch.argmax(alphas[j][1:]) + 1
                            gene.append((PRIMITIVES[idx], j - start))
                else:
                    gene.append((PRIMITIVES[selected_idxs[j]], j - start))
            start = end
            n += 1

        return gene

    def get_genotype(self, force=False):
        if force:
            gene_normal = self.parse_gene_force(self.normal_candidate_flags,
                                                self.normal_selected_idxs,
                                                self.alphas_normal)
        else:
            gene_normal = self.parse_gene(self.normal_selected_idxs)
        n = 2
        concat = range(n + self._steps - self._multiplier, self._steps + n)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat
        )
        return genotype
