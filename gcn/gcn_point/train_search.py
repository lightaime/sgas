import __init__
import os
import sys
import time
import glob
import math
import numpy as np
import torch
from gcn import utils
import logging
import argparse
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as GeoData
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
import torch.backends.cudnn as cudnn
import torch.distributions.categorical as cate
import torchvision.utils as vutils
import sklearn.metrics as metrics
from tqdm import tqdm
from model_search import Network
from architect import Architect
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser("modelnet")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=28, help='batch size')
parser.add_argument('--batch_increase', default=4, type=int,
                    help='how much does the batch size increase after making a decision')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--num_cells', type=int, default=2, help='total number of cells')
parser.add_argument('--n_steps', type=int, default=3, help='total number of layers in one cell')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='ModelNet10', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--random_seed', action='store_true', help='use seed randomly')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--warmup_dec_epoch', type=int, default=9, help='warmup decision epoch')
parser.add_argument('--decision_freq', type=int, default=7, help='decision freq epoch')
parser.add_argument('--history_size', type=int, default=4, help='number of stored epoch scores')
parser.add_argument('--use_history', action='store_true', help='use history for decision')
parser.add_argument('--in_channels', default=3, type=int, help='the channel size of input point cloud ')
parser.add_argument('--emb_dims', default=1024, type=int, help='the channel size of embedding features ')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio')
parser.add_argument('--k', default=9, type=int, help='the kernel size of graph convolutions')
parser.add_argument('--num_points', default=1024, type=int, help='the number of points sampled from modelNet')
parser.add_argument('--post_val', action='store_true', default=False, help='validate after each decision')

args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.save = 'log/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

writer = SummaryWriter(log_dir=args.save, max_queue=50)


def histogram_average(history, probs):
    histogram_inter = torch.zeros(probs.shape[0], dtype=torch.float).cuda()
    if not history:
        return histogram_inter
    for hist in history:
        histogram_inter += utils.histogram_intersection(hist, probs)
    histogram_inter /= len(history)
    return histogram_inter


def score_image(type, score, epoch):
    score_img = vutils.make_grid(
        torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(score, 1), 2), 3),
        nrow=7,
        normalize=True,
        pad_value=0.5)
    writer.add_image(type + '_score', score_img, epoch)


def edge_decision(type, alphas, selected_idxs, candidate_flags, probs_history, epoch, model, args):
    mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
    print(mat)
    importance = torch.sum(mat[:, 1:], dim=-1)
    # logging.info(type + " importance {}".format(importance))

    probs = mat[:, 1:] / importance[:, None]
    # print(type + " probs", probs)
    entropy = cate.Categorical(probs=probs).entropy() / math.log(probs.size()[1])
    # logging.info(type + " entropy {}".format(entropy))

    if args.use_history:  # SGAS Cri.2
        # logging.info(type + " probs history {}".format(probs_history))
        histogram_inter = histogram_average(probs_history, probs)
        # logging.info(type + " histogram intersection average {}".format(histogram_inter))
        probs_history.append(probs)
        if (len(probs_history) > args.history_size):
            probs_history.pop(0)

        score = utils.normalize(importance) * utils.normalize(
            1 - entropy) * utils.normalize(histogram_inter)
        # logging.info(type + " score {}".format(score))
    else:  # SGAS Cri.1
        score = utils.normalize(importance) * utils.normalize(1 - entropy)
        # logging.info(type + " score {}".format(score))

    if torch.sum(candidate_flags.int()) > 0 and \
            epoch >= args.warmup_dec_epoch and \
            (epoch - args.warmup_dec_epoch) % args.decision_freq == 0:
        masked_score = torch.min(score,
                                 (2 * candidate_flags.float() - 1) * np.inf)
        selected_edge_idx = torch.argmax(masked_score)
        selected_op_idx = torch.argmax(probs[selected_edge_idx]) + 1  # add 1 since none op
        selected_idxs[selected_edge_idx] = selected_op_idx

        candidate_flags[selected_edge_idx] = False
        alphas[selected_edge_idx].requires_grad = False
        if type == 'normal':
            reduction = False
        elif type == 'reduce':
            reduction = True
        else:
            raise Exception('Unknown Cell Type')
        candidate_flags, selected_idxs = model.check_edges(candidate_flags,
                                                           selected_idxs)
        logging.info("#" * 30 + " Decision Epoch " + "#" * 30)
        logging.info("epoch {}, {}_selected_idxs {}, added edge {} with op idx {}".format(epoch,
                                                                                          type,
                                                                                          selected_idxs,
                                                                                          selected_edge_idx,
                                                                                          selected_op_idx))
        print(type + "_candidate_flags {}".format(candidate_flags))
        score_image(type, score, epoch)
        return True, selected_idxs, candidate_flags

    else:
        logging.info("#" * 30 + " Not a Decision Epoch " + "#" * 30)
        logging.info("epoch {}, {}_selected_idxs {}".format(epoch,
                                                            type,
                                                            selected_idxs))
        print(type + "_candidate_flags {}".format(candidate_flags))
        score_image(type, score, epoch)
        return False, selected_idxs, candidate_flags

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.random_seed:
        args.seed = np.random.randint(0, 1000, 1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # dataset modelnet
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(args.num_points)
    train_dataset = GeoData.ModelNet(os.path.join(args.data, 'modelnet10'), '10', True, transform, pre_transform)
    train_queue = DenseDataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.batch_size // 2)
    test_dataset = GeoData.ModelNet(os.path.join(args.data, 'modelnet10'), '10', False, transform, pre_transform)
    valid_queue = DenseDataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.batch_size // 2)
    n_classes = train_queue.dataset.num_classes

    criterion = torch.nn.CrossEntropyLoss().cuda()
    model = Network(args.init_channels, n_classes, args.num_cells, criterion,
                    args.n_steps, in_channels=args.in_channels,
                    emb_dims=args.emb_dims, dropout=args.dropout, k=args.k).cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    num_edges = model._steps * 2
    post_train = 5
    args.epochs = args.warmup_dec_epoch + args.decision_freq * (num_edges - 1) + post_train + 1
    logging.info("total epochs: %d", args.epochs)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    normal_selected_idxs = torch.tensor(len(model.alphas_normal) * [-1], requires_grad=False, dtype=torch.int).cuda()
    normal_candidate_flags = torch.tensor(len(model.alphas_normal) * [True], requires_grad=False, dtype=torch.bool).cuda()
    logging.info('normal_selected_idxs: {}'.format(normal_selected_idxs))
    logging.info('normal_candidate_flags: {}'.format(normal_candidate_flags))
    model.normal_selected_idxs = normal_selected_idxs
    model.normal_candidate_flags = normal_candidate_flags

    print(F.softmax(torch.stack(model.alphas_normal, dim=0), dim=-1).detach())

    count = 0
    normal_probs_history = []
    train_losses, valid_losses = utils.AverageMeter(), utils.AverageMeter()
    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_losses = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, train_losses)
        valid_overall_acc, valid_class_acc, valid_losses = infer(valid_queue, model, criterion, valid_losses)

        logging.info('train_acc %f\tvalid_overall_acc %f \t valid_class_acc %f',
                     train_acc, valid_overall_acc, valid_class_acc)

        # make edge decisions
        saved_memory_normal, model.normal_selected_idxs, \
        model.normal_candidate_flags = edge_decision('normal',
                                                     model.alphas_normal,
                                                     model.normal_selected_idxs,
                                                     model.normal_candidate_flags,
                                                     normal_probs_history,
                                                     epoch,
                                                     model,
                                                     args)

        if saved_memory_normal:
            del train_queue, valid_queue
            torch.cuda.empty_cache()

            count += 1
            new_batch_size = args.batch_size + args.batch_increase * count
            logging.info("new_batch_size = {}".format(new_batch_size))
            train_queue = DenseDataLoader(train_dataset, batch_size=new_batch_size, shuffle=True,
                                          num_workers=args.batch_size // 2)
            valid_queue = DenseDataLoader(test_dataset, batch_size=new_batch_size, shuffle=False,
                                          num_workers=args.batch_size // 2)
            # post validation
            if args.post_val:
                post_valid_overall_acc, post_valid_class_acc, valid_losses = infer(valid_queue, model, criterion, valid_losses)
                logging.info('post_valid_overall_acc %f', post_valid_overall_acc)

        writer.add_scalar('stats/train_acc', train_acc, epoch)
        writer.add_scalar('stats/valid_overall_acc', valid_overall_acc, epoch)
        writer.add_scalar('stats/valid_class_acc', valid_class_acc, epoch)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        scheduler.step()

    logging.info("#" * 30 + " Done " + "#" * 30)
    logging.info('genotype = %s', model.get_genotype())


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, train_losses):
    # reset loss tracker
    test_true = []
    test_pred = []
    train_losses.reset()

    for step, data in enumerate(tqdm(train_queue)):
        model.train()
        input = data.pos.transpose(2, 1).unsqueeze(3)
        input = input.to(DEVICE)
        target = data.y.squeeze().to(DEVICE)
        n = input.size(0)

        data_search = next(iter(valid_queue))
        input_search = data_search.pos.transpose(2, 1).unsqueeze(3).to(DEVICE)
        target_search = data_search.y.squeeze().to(DEVICE)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        out = model(input)
        loss = criterion(out, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        pred = out.max(dim=1)[1]
        test_true.append(target.cpu().numpy())
        test_pred.append(pred.detach().cpu().numpy())
        train_losses.update(loss.item(), n)

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    return test_acc, train_losses


def infer(valid_queue, model, criterion, valid_losses):
    model.eval()
    test_true = []
    test_pred = []
    valid_losses.reset()

    with torch.no_grad():
        for i, data in enumerate(valid_queue):
            data = data.to(DEVICE)
            inputs = data.pos.transpose(2, 1).unsqueeze(3)
            gt = data.y
            out = model(inputs)

            pred = out.max(dim=1)[1]
            test_true.append(gt.cpu().numpy())
            test_pred.append(pred.detach().cpu().numpy())
            loss = criterion(out, gt.squeeze())

            n = gt.size(0)
            valid_losses.update(loss.item(), n)

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        overall_acc = metrics.accuracy_score(test_true, test_pred)
        class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    return overall_acc, class_acc, valid_losses


if __name__ == '__main__':
    main()
