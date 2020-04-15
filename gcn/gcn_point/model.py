import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPS, BasicConv, Identity
from gcn.gcn_lib.dense import DilatedKnn2d
from gcn.utils import drop_path


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, k=9, d=1):
        super(Cell, self).__init__()
        self.preprocess0 = BasicConv([C_prev_prev, C], 'relu', 'batch', bias=False)
        self.preprocess1 = BasicConv([C_prev, C], 'relu', 'batch', bias=False)
        self.dilated_knn_graph = DilatedKnn2d(k=k, dilation=d)

        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C, 1, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        edge_index = self.dilated_knn_graph(s0)
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1, edge_index)
            h2 = op2(h2, edge_index)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHead(nn.Module):

    def __init__(self, C, num_classes):
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(
            BasicConv([C, 128, 768], 'relu', 'batch', bias=False)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class NetworkModelNet40(nn.Module):

    def __init__(self, C, num_classes, num_cells, auxiliary, genotype, stem_multiplier=3,
                 in_channels=3, emb_dims=1024, dropout=0.5, k=9):
        super(NetworkModelNet40, self).__init__()
        self._layers = num_cells
        self._auxiliary = auxiliary
        self._in_channels = in_channels
        self.drop_path_prob = 0.
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            BasicConv([in_channels, C_curr], None, 'batch', bias=False),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        for i in range(self._layers):
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, k=k, d=i + 1)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * self._layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes)
        self.fusion_conv = BasicConv([stem_multiplier * C + C * cell.multiplier * self._layers, emb_dims],
                                     act='leakyrelu', norm='batch', bias=False)
        self.classifier = nn.Sequential(BasicConv([emb_dims * 2, 512], act='leakyrelu', norm='batch'),
                                        torch.nn.Dropout(p=dropout),
                                        BasicConv([512, 256], act='leakyrelu', norm='batch'),
                                        torch.nn.Dropout(p=dropout),
                                        BasicConv([256, num_classes], act=None, norm=None))

    def forward(self, x):
        logits_aux = None
        s0 = s1 = self.stem(x)
        pre_layers = [s1]
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            pre_layers.append(s1)

            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1).squeeze(-1).squeeze(-1)

        fusion = torch.cat(pre_layers, dim=1)
        fusion = self.fusion_conv(fusion)
        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        logits = self.classifier(torch.cat((x1, x2), dim=1)).squeeze(-1).squeeze(-1)
        return logits, logits_aux

