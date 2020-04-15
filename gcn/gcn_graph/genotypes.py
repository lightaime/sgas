from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')
Genotype_normal = namedtuple('Genotype_normal', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_1x1',
    'edge_conv',
    'mr_conv',
    'gat',
    'semi_gcn',
    'gin',
    'sage',
    'res_sage',
]

# ****************************  SGAS CRITERION 1  ****************************** #
# search train acc	| search val acc | 	params	|   best test acc | Evaluation Ranking
Cri1_PPI_1 = Genotype(normal=[('edge_conv', 0), ('mr_conv', 1), ('res_sage', 0), ('edge_conv', 2), ('edge_conv', 1), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri1_PPI_2 = Genotype(normal=[('edge_conv', 0), ('gat', 1), ('res_sage', 0), ('edge_conv', 2), ('edge_conv', 2), ('skip_connect', 3)], normal_concat=range(1, 5))
Cri1_PPI_3 = Genotype(normal=[('sage', 0), ('gin', 1), ('gat', 1), ('edge_conv', 2), ('sage', 2), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri1_PPI_4 = Genotype(normal=[('edge_conv', 0), ('edge_conv', 1), ('res_sage', 0), ('gin', 2), ('gat', 1), ('res_sage', 3)], normal_concat=range(1, 5))
Cri1_PPI_5 = Genotype(normal=[('gat', 0), ('mr_conv', 1), ('edge_conv', 0), ('semi_gcn', 1), ('sage', 0), ('edge_conv', 2)], normal_concat=range(1, 5))
Cri1_PPI_6 = Genotype(normal=[('semi_gcn', 0), ('sage', 1), ('sage', 0), ('res_sage', 2), ('res_sage', 2), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri1_PPI_7 = Genotype(normal=[('gat', 0), ('edge_conv', 1), ('edge_conv', 1), ('res_sage', 2), ('gat', 0), ('mr_conv', 1)], normal_concat=range(1, 5))
Cri1_PPI_8 = Genotype(normal=[('res_sage', 0), ('edge_conv', 1), ('semi_gcn', 0), ('res_sage', 2), ('skip_connect', 2), ('res_sage', 3)], normal_concat=range(1, 5))
Cri1_PPI_9 = Genotype(normal=[('gat', 0), ('edge_conv', 1), ('semi_gcn', 0), ('sage', 1), ('conv_1x1', 2), ('res_sage', 3)], normal_concat=range(1, 5))
Cri1_PPI_10 = Genotype(normal=[('edge_conv', 0), ('mr_conv', 1), ('mr_conv', 0), ('gin', 1), ('semi_gcn', 2), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri1_PPI_Best = Cri1_PPI_10

# ****************************  SGAS CRITERION 2  ****************************** #
Cri2_PPI_1 = Genotype(normal=[('sage', 0), ('gin', 1), ('res_sage', 1), ('edge_conv', 2), ('gat', 1), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri2_PPI_2 = Genotype(normal=[('res_sage', 0), ('mr_conv', 1), ('edge_conv', 0), ('edge_conv', 1), ('res_sage', 1), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri2_PPI_3 = Genotype(normal=[('res_sage', 0), ('edge_conv', 1), ('gat', 0), ('skip_connect', 2), ('edge_conv', 1), ('skip_connect', 2)], normal_concat=range(1, 5))
Cri2_PPI_4 = Genotype(normal=[('sage', 0), ('semi_gcn', 1), ('gin', 0), ('gin', 1), ('gat', 2), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri2_PPI_5 = Genotype(normal=[('edge_conv', 0), ('semi_gcn', 1), ('gat', 0), ('edge_conv', 1), ('edge_conv', 2), ('sage', 3)], normal_concat=range(1, 5))
Cri2_PPI_6 = Genotype(normal=[('mr_conv', 0), ('sage', 1), ('semi_gcn', 0), ('edge_conv', 2), ('res_sage', 0), ('sage', 2)], normal_concat=range(1, 5))
Cri2_PPI_7 = Genotype(normal=[('sage', 0), ('edge_conv', 1), ('sage', 1), ('gin', 2), ('sage', 0), ('sage', 1)], normal_concat=range(1, 5))
Cri2_PPI_8 = Genotype(normal=[('res_sage', 0), ('mr_conv', 1), ('gin', 0), ('sage', 2), ('gat', 1), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri2_PPI_9 = Genotype(normal=[('edge_conv', 0), ('sage', 1), ('gat', 0), ('sage', 2), ('res_sage', 2), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri2_PPI_10 = Genotype(normal=[('mr_conv', 0), ('skip_connect', 1), ('res_sage', 0), ('semi_gcn', 2), ('res_sage', 2), ('res_sage', 3)], normal_concat=range(1, 5))
Cri2_PPI_Best = Cri2_PPI_2

# ****************************  Random Search   ****************************** #
Random_PPI_1 = Genotype(normal=[('gat', 0), ('res_sage', 1), ('semi_gcn', 2), ('semi_gcn', 1), ('gat', 2), ('gat', 3)], normal_concat=range(1, 5))
Random_PPI_2 = Genotype(normal=[('skip_connect', 0), ('sage', 1), ('sage', 2), ('gin', 0), ('mr_conv', 1), ('mr_conv', 0)], normal_concat=range(1, 5))
Random_PPI_3 = Genotype(normal=[('conv_1x1', 0), ('gin', 1), ('semi_gcn', 1), ('mr_conv', 2), ('gin', 3), ('skip_connect', 2)], normal_concat=range(1, 5))
Random_PPI_4 = Genotype(normal=[('conv_1x1', 0), ('skip_connect', 1), ('conv_1x1', 2), ('gin', 1), ('sage', 0), ('semi_gcn', 3)], normal_concat=range(1, 5))
Random_PPI_5 = Genotype(normal=[('mr_conv', 1), ('mr_conv', 0), ('res_sage', 1), ('semi_gcn', 2), ('sage', 1), ('skip_connect', 2)], normal_concat=range(1, 5))
Random_PPI_6 = Genotype(normal=[('res_sage', 0), ('sage', 1), ('sage', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 0)], normal_concat=range(1, 5))
Random_PPI_7 = Genotype(normal=[('conv_1x1', 1), ('sage', 0), ('sage', 1), ('mr_conv', 0), ('sage', 2), ('semi_gcn', 0)], normal_concat=range(1, 5))
Random_PPI_8 = Genotype(normal=[('mr_conv', 1), ('gin', 0), ('skip_connect', 1), ('conv_1x1', 2), ('res_sage', 2), ('res_sage', 0)], normal_concat=range(1, 5))
Random_PPI_9 = Genotype(normal=[('skip_connect', 1), ('gin', 0), ('sage', 2), ('gin', 1), ('sage', 1), ('sage', 0)], normal_concat=range(1, 5))
Random_PPI_10 = Genotype(normal=[('sage', 0), ('mr_conv', 1), ('res_sage', 1), ('conv_1x1', 0), ('sage', 2), ('gat', 1)], normal_concat=range(1, 5))
Random_PPI_best = Random_PPI_8

