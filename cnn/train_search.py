import os
import sys
import time
import glob
import math
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.distributions.categorical as cate
import torchvision.utils as vutils

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--batch_increase', default=8, type=int, help='how much does the batch size increase after making a decision')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--warmup_dec_epoch', type=int, default=9, help='warmup decision epoch')
parser.add_argument('--decision_freq', type=int, default=5, help='decision freq epoch')
parser.add_argument('--use_history', action='store_true', help='use history for decision')
parser.add_argument('--history_size', type=int, default=4, help='number of stored epoch scores')
parser.add_argument('--post_val', action='store_true', default=False, help='validate after each decision')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

writer = SummaryWriter(log_dir=args.save, max_queue=50)

CIFAR_CLASSES = 10

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

  if args.use_history: # SGAS Cri.2
    # logging.info(type + " probs history {}".format(probs_history))
    histogram_inter = histogram_average(probs_history, probs)
    # logging.info(type + " histogram intersection average {}".format(histogram_inter))
    probs_history.append(probs)
    if (len(probs_history) > args.history_size):
      probs_history.pop(0)

    score = utils.normalize(importance) * utils.normalize(
      1 - entropy) * utils.normalize(histogram_inter)
    # logging.info(type + " score {}".format(score))
  else: # SGAS Cri.1
    score = utils.normalize(importance) * utils.normalize(1 - entropy)
    # logging.info(type + " score {}".format(score))

  if torch.sum(candidate_flags.int()) > 0 and \
      epoch >= args.warmup_dec_epoch and \
      (epoch - args.warmup_dec_epoch) % args.decision_freq == 0:
    masked_score = torch.min(score,
                              (2 * candidate_flags.float() - 1) * np.inf)
    selected_edge_idx = torch.argmax(masked_score)
    selected_op_idx = torch.argmax(probs[selected_edge_idx]) + 1 # add 1 since none op
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
                                                       selected_idxs,
                                                       reduction=reduction)
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

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  num_edges = model._steps * 2
  post_train = 5
  epochs = args.warmup_dec_epoch + args.decision_freq * (num_edges - 1) + post_train + 1
  logging.info("total epochs: %d", epochs)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  normal_selected_idxs = torch.tensor(len(model.alphas_normal) * [-1], requires_grad=False, dtype=torch.int).cuda()
  reduce_selected_idxs = torch.tensor(len(model.alphas_reduce) * [-1], requires_grad=False, dtype=torch.int).cuda()
  normal_candidate_flags = torch.tensor(len(model.alphas_normal) * [True], requires_grad=False, dtype=torch.bool).cuda()
  reduce_candidate_flags = torch.tensor(len(model.alphas_reduce) * [True], requires_grad=False, dtype=torch.bool).cuda()
  logging.info('normal_selected_idxs: {}'.format(normal_selected_idxs))
  logging.info('reduce_selected_idxs: {}'.format(reduce_selected_idxs))
  logging.info('normal_candidate_flags: {}'.format(normal_candidate_flags))
  logging.info('reduce_candidate_flags: {}'.format(reduce_candidate_flags))
  model.normal_selected_idxs = normal_selected_idxs
  model.reduce_selected_idxs = reduce_selected_idxs
  model.normal_candidate_flags = normal_candidate_flags
  model.reduce_candidate_flags = reduce_candidate_flags

  print(F.softmax(torch.stack(model.alphas_normal, dim=0), dim=-1).detach())
  print(F.softmax(torch.stack(model.alphas_reduce, dim=0), dim=-1).detach())

  count = 0
  normal_probs_history = []
  reduce_probs_history = []

  for epoch in range(epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    with torch.no_grad():
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)


    saved_memory_normal, model.normal_selected_idxs, \
    model.normal_candidate_flags = edge_decision('normal',
                                                 model.alphas_normal,
                                                 model.normal_selected_idxs,
                                                 model.normal_candidate_flags,
                                                 normal_probs_history,
                                                 epoch,
                                                 model,
                                                 args)

    saved_memory_reduce, model.reduce_selected_idxs, \
    model.reduce_candidate_flags = edge_decision('reduce',
                                                 model.alphas_reduce,
                                                 model.reduce_selected_idxs,
                                                 model.reduce_candidate_flags,
                                                 reduce_probs_history,
                                                 epoch,
                                                 model,
                                                 args)

    if saved_memory_normal or saved_memory_reduce:
      del train_queue, valid_queue
      torch.cuda.empty_cache()

      count += 1
      new_batch_size = args.batch_size + args.batch_increase * count
      logging.info("new_batch_size = {}".format(new_batch_size))
      train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=new_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

      valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=new_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

      # post validation
      if args.post_val:
        with torch.no_grad():
          post_valid_acc, valid_obj = infer(valid_queue, model, criterion)
          logging.info('post_valid_acc %f', post_valid_acc)

    logging.info('genotype = %s', model.get_genotype(force=True))
    writer.add_scalar('stats/train_acc', train_acc, epoch)
    writer.add_scalar('stats/valid_acc', valid_acc, epoch)
    utils.save(model, os.path.join(args.save, 'weights.pt'))

  logging.info("#" * 30 + " Done " + "#" * 30)
  logging.info('genotype = %s', model.get_genotype())


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

