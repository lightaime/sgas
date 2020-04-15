import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import f1_score
import logging


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def average_mF1(model, loader, opt):
    model.eval()
    count = 0
    micro_f1 = 0.
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(opt.device)
            out = model(data)

            num_node = len(data.x)
            micro_f1 += f1_score(data.y.cpu().detach().numpy(),
                                 (out > 0).cpu().detach().numpy(), average='micro') * num_node
            count += num_node
        micro_f1 = float(micro_f1) / count
    return micro_f1


def mF1(output, target):
    micro_f1 = f1_score(target.cpu().detach().numpy(),
                        (output > 0).cpu().detach().numpy(), average='micro')

    return micro_f1


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def normalize(v):
    min_v = torch.min(v)
    range_v = torch.max(v) - min_v
    if range_v > 0:
        normalized_v = (v - min_v) / range_v
    else:
        normalized_v = torch.zeros(v.size()).cuda()

    return normalized_v


def histogram_intersection(a, b):
    c = np.minimum(a.cpu().numpy(), b.cpu().numpy())
    c = torch.from_numpy(c).cuda()
    sums = c.sum(dim=1)
    return sums


def translate_pointcloud(pointcloud):
    scale = torch.FloatTensor(3).uniform_(2. / 3., 3. / 2.)
    offset = torch.FloatTensor(3).uniform_(-0.2, 0.2)
    translated_pointcloud = torch.mul(pointcloud, scale) + offset
    return translated_pointcloud


def load_pretrained_models(model, pretrained_model, phase, ismax=True):  # ismax means max best
    if ismax:
        best_value = -np.inf
    else:
        best_value = np.inf
    epoch = -1

    if pretrained_model:
        if os.path.isfile(pretrained_model):
            logging.info("===> Loading checkpoint '{}'".format(pretrained_model))
            checkpoint = torch.load(pretrained_model)
            try:
                best_value = checkpoint['best_value']
                if best_value == -np.inf or best_value == np.inf:
                    show_best_value = False
                else:
                    show_best_value = True
            except:
                best_value = best_value
                show_best_value = False

            model_dict = model.state_dict()
            ckpt_model_state_dict = checkpoint['state_dict']

            # rename ckpt (avoid name is not same because of multi-gpus)
            is_model_multi_gpus = True if list(model_dict)[0][0][0] == 'm' else False
            is_ckpt_multi_gpus = True if list(ckpt_model_state_dict)[0][0] == 'm' else False

            if not (is_model_multi_gpus == is_ckpt_multi_gpus):
                temp_dict = OrderedDict()
                for k, v in ckpt_model_state_dict.items():
                    if is_ckpt_multi_gpus:
                        name = k[7:]  # remove 'module.'
                    else:
                        name = 'module.' + k  # add 'module'
                    temp_dict[name] = v
                # load params
                ckpt_model_state_dict = temp_dict

            model_dict.update(ckpt_model_state_dict)
            model.load_state_dict(ckpt_model_state_dict)

            if show_best_value:
                logging.info("The pretrained_model is at checkpoint {}. \t "
                             "Best value: {}".format(checkpoint['epoch'], best_value))
            else:
                logging.info("The pretrained_model is at checkpoint {}.".format(checkpoint['epoch']))

            if phase == 'train':
                epoch = checkpoint['epoch']
            else:
                epoch = -1
        else:
            raise ImportError("===> No checkpoint found at '{}'".format(pretrained_model))
    else:
        logging.info('===> No pre-trained model')
    return model, best_value, epoch


def load_pretrained_optimizer(pretrained_model, optimizer, scheduler, lr, use_ckpt_lr=True):
    if pretrained_model:
        if os.path.isfile(pretrained_model):
            checkpoint = torch.load(pretrained_model)
            if 'optimizer_state_dict' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            if 'scheduler_state_dict' in checkpoint.keys():
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if use_ckpt_lr:
                    try:
                        lr = scheduler.get_lr()[0]
                    except:
                        lr = lr

    return optimizer, scheduler, lr

