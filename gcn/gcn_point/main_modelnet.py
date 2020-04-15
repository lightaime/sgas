import __init__
import os
import sys
import time
import glob
import numpy as np
import torch
from gcn import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from model import NetworkModelNet40 as Network
import sklearn.metrics as metrics
from tqdm import tqdm
from load_modelnet import ModelNet40
from torch.utils.data import DataLoader
import uuid
# this is used for loading cells for evaluation
import genotypes

parser = argparse.ArgumentParser("modelnet")

parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--phase', type=str, default='train', help='train/test')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')
parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=400, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=128, help='num of init channels')
parser.add_argument('--num_cells', type=int, default=9, help='total number of cells')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model / or pretrained')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--data_augment', action='store_true', default=True, help='data_augment')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=4, help='random seed')
parser.add_argument('--arch', type=str, default='Cri2_ModelNet_Best', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--in_channels', default=3, type=int, help='the channel size of data point cloud ')
parser.add_argument('--emb_dims', default=1024, type=int, help='the channel size of embedding features ')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio')
parser.add_argument('--k', default=20, type=int, help='the kernel size of graph convolutions')
parser.add_argument('--num_points', default=1024, type=int, help='the number of points sampled from modelNet')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--hyper_search', action='store_true', default=False, help='try different seed')

args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.save = 'log/{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), str(uuid.uuid4()))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def train():
    best_test_acc = 0.
    class_acc_best = 0.
    for epoch in range(args.epochs):
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_overall_acc, train_class_acc, train_obj = train_step(train_queue, model, criterion, optimizer, args)
        test_overall_acc, test_class_acc, test_obj = infer(test_queue, model, criterion)
        if test_overall_acc > best_test_acc:
            best_test_acc = test_overall_acc
            class_acc_best = test_class_acc
            utils.save(model, os.path.join(args.save, 'best_weights.pt'))

        logging.info(
            'train_overall_acc %f\t train_class_acc %f\t test_overall_acc %f\t test_class_acc %f\t best_test_overall_acc %f\t test_class_acc_when_best %f',
            train_overall_acc, train_class_acc, test_overall_acc, test_class_acc, best_test_acc, class_acc_best)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        scheduler.step()

    logging.info(
        'Finish! best_test_overall_acc %f\t test_class_acc_when_best %f', best_test_acc, class_acc_best)


def train_step(train_queue, model, criterion, optimizer, args):
    objs = utils.AverageMeter()
    train_true = []
    train_pred = []
    for step, (data, label) in enumerate(tqdm(train_queue)):
        model.train()
        data, label = data.to(DEVICE), label.to(DEVICE).squeeze()
        data = data.permute(0, 2, 1).unsqueeze(3)
        n = data.size(0)

        optimizer.zero_grad()
        out, out_aux = model(data)
        loss = criterion(out, label)
        if args.auxiliary:
            loss_aux = criterion(out_aux, label)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        pred = out.max(dim=1)[1]
        train_true.append(label.cpu().numpy())
        train_pred.append(pred.detach().cpu().numpy())
        objs.update(loss.item(), n)

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    overall_acc = metrics.accuracy_score(train_true, train_pred)
    class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
    return overall_acc, class_acc, objs.avg


def infer(test_queue, model, criterion):
    model.eval()
    objs = utils.AverageMeter()
    test_true = []
    test_pred = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_queue):
            data, label = data.to(DEVICE), label.to(DEVICE).squeeze()
            data = data.permute(0, 2, 1).unsqueeze(3)

            out, out_aux = model(data)

            pred = out.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred.detach().cpu().numpy())
            loss = criterion(out, label.squeeze())

            n = label.size(0)
            objs.update(loss.item(), n)

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        overall_acc = metrics.accuracy_score(test_true, test_pred)
        class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    return overall_acc, class_acc, objs.avg


if __name__ == '__main__':
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.hyper_search:
        args.seed = np.random.randint(0, 1000, 1)[0]
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # dataset modelnet
    train_queue = DataLoader(ModelNet40(partition='train', num_points=args.num_points, data_dir=args.data),
                             num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_queue = DataLoader(ModelNet40(partition='test', num_points=args.num_points, data_dir=args.data),
                            num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False)

    n_classes = 40
    logging.info('n_classes: %d', n_classes)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, n_classes, args.num_cells, args.auxiliary, genotype,
                    in_channels=args.in_channels,
                    emb_dims=args.emb_dims, dropout=args.dropout, k=args.k)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.use_sgd:
        print("Use SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate * 100,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Use Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate)

    logging.info('phase is {}'.format(args.phase))
    if args.phase == 'test':
        logging.info("===> Loading checkpoint '{}'".format(args.model_path))
        utils.load(model, args.model_path)
        test_overall_acc, test_class_acc, test_obj = infer(test_queue, model, criterion)
        logging.info(
            'Finish Testing! test_overall_acc %f\t test_class_acc %f', test_overall_acc, test_class_acc)
    else:
        train()

