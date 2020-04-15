import argparse
import numpy as np
import torch
import torch.nn.functional as F
from genotypes import PRIMITIVES
from model_search import Network
from torch.autograd import Variable
from collections import namedtuple


def random_alphas(steps):
    k = sum(1 for i in range(steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)

    alphas_normal = []
    for i in range(steps):
        for n in range(2 + i):
            alphas_normal.append(Variable(torch.randn(1, num_ops).cuda(), requires_grad=True))
    return alphas_normal


def normalize_weights(alphas, steps):
    normal_weights = []
    n = 2
    start = 0
    for _ in range(steps):
        end = start + n
        for j in range(start, end):
            normal_weights.append(F.softmax(alphas[j], dim=-1).data.cpu().numpy())
        start = end
        n += 1
    return normal_weights


def parse(weights, steps):
    gene = []
    n = 2
    start = 0
    for i in range(steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
            k_best = None
            for k in range(len(W[j])):
                if k != PRIMITIVES.index('none'):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
    multiplier = 4
    concat = range(2 + steps - multiplier, steps + 2)
    Genotype = namedtuple('Genotype', 'normal normal_concat')
    genotype_normal = Genotype(normal=gene, normal_concat=concat)
    return genotype_normal


parser = argparse.ArgumentParser("ppi")
parser.add_argument('--n_steps', type=int, default=3, help='total number of layers in one cell')
parser.add_argument('--n_classes', type=int, default=121, help='total number of classes')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--in_channels', type=int, default=50, help='num of init channels')
parser.add_argument('--num_cells', type=int, default=1, help='total number of cells')
parser.add_argument('--n_archs', type=int, default=1, help='Number of random architectures to output')

args = parser.parse_args()

criterion = torch.nn.BCEWithLogitsLoss().cuda()
model = Network(args.init_channels, args.n_classes, args.num_cells, criterion,
                args.n_steps, in_channels=args.in_channels).cuda()
alphas_normal = torch.cat(model.alphas_normal, dim=0)

num_opts = len(PRIMITIVES)
opt_random_idx = np.random.randint(0, num_opts - 1, 2 * args.n_steps)
connection_random_idx = np.random.randint(0, args.n_steps, 2 * args.n_steps)
for i in range(args.n_archs):
    alphas_normal = random_alphas(args.n_steps)
    normal_weights = normalize_weights(alphas_normal, args.n_steps)
    gene_normal = parse(np.concatenate(normal_weights, axis=0), args.n_steps)

    print(gene_normal)
