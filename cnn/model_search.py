import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np

class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights, selected_idx=None):
        if selected_idx is None:
            return sum(w * op(x) for w, op in zip(weights, self._ops))
        else:  # unchosen operations are pruned
            return self._ops[selected_idx](x)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, selected_idxs=None):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            o_list = []
            for j, h in enumerate(states):
                if selected_idxs[offset + j] == -1: # undecided mix edges
                    o = self._ops[offset + j](h, weights[offset + j])
                elif selected_idxs[offset + j] == PRIMITIVES.index('none'): # pruned edges
                    pass
                else: # decided discrete edges
                    o = self._ops[offset + j](h, None, selected_idxs[offset + j])
                o_list.append(o)
            s = sum(o_list)
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

        self.normal_selected_idxs = None
        self.reduce_selected_idxs = None
        self.normal_candidate_flags = None
        self.reduce_candidate_flags = None

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        model_new.normal_selected_idxs = self.normal_selected_idxs
        model_new.reduce_selected_idxs = self.reduce_selected_idxs
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                selected_idxs = self.reduce_selected_idxs
                alphas = self.alphas_reduce
            else:
                selected_idxs = self.normal_selected_idxs
                alphas = self.alphas_normal

            weights = []
            n = 2
            start = 0
            for _ in range(self._steps):
                end = start + n
                for j in range(start, end):
                    weights.append(F.softmax(alphas[j], dim=-1))
                start = end
                n += 1

            s0, s1 = s1, cell(s0, s1, weights, selected_idxs)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = []
        self.alphas_reduce = []
        for i in range(self._steps):
            for n in range(2 + i):
                self.alphas_normal.append(Variable(1e-3 * torch.randn(num_ops).cuda(), requires_grad=True))
                self.alphas_reduce.append(Variable(1e-3 * torch.randn(num_ops).cuda(), requires_grad=True))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self.alphas_normal + self.alphas_reduce # concat lists

    def check_edges(self, flags, selected_idxs, reduction=False):
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
                        if reduction:
                            self.alphas_reduce[j].requires_grad = False
                        else:
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
          gene_reduce = self.parse_gene_force(self.reduce_candidate_flags,
                                              self.reduce_selected_idxs,
                                              self.alphas_reduce)
        else:
          gene_normal = self.parse_gene(self.normal_selected_idxs)
          gene_reduce = self.parse_gene(self.reduce_selected_idxs)
        n = 2
        concat = range(n + self._steps - self._multiplier, self._steps + n)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
