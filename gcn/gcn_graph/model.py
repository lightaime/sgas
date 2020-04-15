import __init__
import torch
from operations import *
from gcn.utils import drop_path


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C):
        super(Cell, self).__init__()
        self.preprocess0 = MLP([C_prev_prev, C], 'relu', 'batch', bias=False)
        self.preprocess1 = MLP([C_prev, C], 'relu', 'batch', bias=False)

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

    def forward(self, s0, s1, edge_index, drop_prob):
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


class AuxiliaryHeadPPI(nn.Module):

    def __init__(self, C, num_classes):
        super(AuxiliaryHeadPPI, self).__init__()
        self.features = nn.Sequential(
            MLP([C, 128, 768], 'relu', 'batch', bias=False)

        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class NetworkPPI(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, stem_multiplier=3, in_channels=3):
        super(NetworkPPI, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self._in_channels = in_channels
        self.drop_path_prob = 0.

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            MLP([in_channels, C_curr], None, 'batch', bias=False),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadPPI(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C_prev + 1, num_classes)

    def forward(self, input):
        logits_aux = None
        x, edge_index = input.x, input.edge_index
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, edge_index, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1.unsqueeze(0)).squeeze(0)
        logits = self.classifier(torch.cat((out, s1), dim=1))
        return logits, logits_aux
