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
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader
import torch.utils
import torch.backends.cudnn as cudnn
from model import NetworkPPI as Network
# this is used for loading cells for evaluation
import genotypes


parser = argparse.ArgumentParser("ppi")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--phase', type=str, default='train', help='train/test')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.002, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=2000, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=512, help='num of init channels')
parser.add_argument('--num_cells', type=int, default=5, help='total number of cells')
parser.add_argument('--model_path', type=str, default='log/ckpt', help='path to save the model / pretrained')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--arch', type=str, default='Cri2_PPI_Best', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--in_channels', default=50, type=int, help='the channel size of input point cloud ')
args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.save = 'log/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def train():
    best_val_acc = 0.
    best_test_acc = 0.
    for epoch in range(args.epochs):
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train_step(train_queue, model, criterion, optimizer)
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        test_acc, test_obj = infer(test_queue, model, criterion)
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            test_acc_when_best_val = test_acc
            utils.save(model, os.path.join(args.save, 'best_weights.pt'))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        logging.info('train_acc %f\tvalid_acc %f\tbest_val_acc %f\ttest_acc %f\tbest_test_acc %f\tfinal_best_test %f',
                     train_acc, valid_acc, best_val_acc, test_acc, best_test_acc, test_acc_when_best_val)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        scheduler.step()
    logging.info(
        'Finish! best_val_acc %f\t test_class_acc_when_best %f \t best test %f',
        best_test_acc, test_acc_when_best_val, best_test_acc)


def train_step(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    micro_f1 = 0.
    count = 0.
    for step, input in enumerate(train_queue):
        model.train()
        input = input.to(DEVICE)
        target = input.y
        n = input.x.size(0)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        micro_f1 += utils.mF1(logits, target) * n
        count += n
        objs.update(loss.item(), n)
    micro_f1 = float(micro_f1) / count
    return micro_f1, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    model.eval()
    count = 0.
    micro_f1 = 0.

    with torch.no_grad():
        for step, input in enumerate(valid_queue):
            input = input.to(DEVICE)
            target = input.y
            logits, _ = model(input)
            loss = criterion(logits, target)

            n = target.size(0)
            micro_f1 += utils.mF1(logits, target) * n
            count += n
            objs.update(loss.item(), n)
    micro_f1 = float(micro_f1) / count
    return micro_f1, objs.avg


if __name__ == '__main__':
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    train_dataset = GeoData.PPI(os.path.join(args.data, 'ppi'), split='train')
    train_queue = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = GeoData.PPI(os.path.join(args.data, 'ppi'), split='val')
    valid_queue = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = GeoData.PPI(os.path.join(args.data, 'ppi'), split='test')
    test_queue = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    n_classes = train_queue.dataset.num_classes

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, n_classes, args.num_cells, args.auxiliary, genotype,
                    in_channels=args.in_channels)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    if args.phase == 'test':
        logging.info("===> Loading checkpoint '{}'".format(args.model_path))
        utils.load(model, args.model_path)
        test_acc, test_obj = infer(test_queue, model, criterion)
        logging.info(
            'Finish Testing! test_acc %f', test_acc)
    else:
        train()

