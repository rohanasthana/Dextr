'''
    main process for train a bulk of networks
'''
#!/usr/bin/python
#!python
from distutils.command.build import build
import os
import pdb
import time
import pickle
import random
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from thop import profile
import json

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# from models import model_dict
import models
from dag_utils import build_model
from dataset import cifar10_dataloaders, cifar100_dataloaders, svhn_dataloaders, mnist_dataloaders, imagenet_dataloaders
from logger import prepare_seed, prepare_logger
from utils import save_checkpoint, warmup_lr, AverageMeter, accuracy

from pruner import check_sparsity, pruning_model_random, pruning_model_random_layer_specified, prune_model_custom, extract_mask

from pdb import set_trace as bp

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training Subnetworks')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='mlp', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--dag', type=str, default='', help='from-to edges separated by underscore. 0: broken edge; 1: skip-connect; 2: linear or conv')
parser.add_argument('--imagenet_arch', action="store_true", help="back to imagenet architecture (conv1, maxpool)")
parser.add_argument('--width', type=int, default=2048, help='hidden width')
parser.add_argument('--bn', action="store_true", help="use BN")
parser.add_argument('--fc', action="store_true", help="use FC in ConvNet")
parser.add_argument('--rand_prune', default=-1, type=float, help="random global unstructured: fraction of parameters to prune")

##################################### General setting ############################################
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--inference', action="store_true", help="testing")
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='./experiment', type=str)
parser.add_argument('--exp_name', help='additional names for experiment', default='', type=str)
parser.add_argument('--repeat', default=1, type=int, help='repeat training of DAG w. different random seed')
parser.add_argument('--reverse_order', action="store_true", help="bulk train in reverse order")
parser.add_argument('--start_idx', type=int, default=-1, help='index of first dag to train (inclusive)')
parser.add_argument('--end_idx', type=int, default=-1, help='index of last dag to train (EXclusive)')

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0., type=float, help='momentum') # 0.9
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay') # 1e-4
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
# parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')
parser.add_argument('--decreasing_lr', default=None, help='decreasing strategy')
parser.add_argument('--save_ckeckpoint_freq', default=-1, type=int, help='save intermediate checkpoint per epoch')
parser.add_argument('--pretrained', action="store_true", help="use official pretrained checkpoint")


# best_acc = 0
args = parser.parse_args()

if args.seed is None:
    args.seed = random.randint(0, 999)


ING_GAP_SEC = 240 # if an ".ing" file is idled for over ING_GAP_SEC second, then this job is killed, can resume; otherwise, there is still a running job
def check_modified_time(filename):
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(filename)
    return mtime


def main():
    global args
    # global best_acc
    torch.cuda.set_device(int(args.gpu))

    if args.dataset in ['cifar10', 'cifar100']:
        from torchvision.datasets import CIFAR10, CIFAR100
        CIFAR10(args.data, train=True, download=True)
        CIFAR10(args.data, train=False, download=True)
        CIFAR100(args.data, train=True, download=True)
        CIFAR100(args.data, train=False, download=True)

    job_name = "BULK-{dataset}-{arch}{width}{bn}-LR{lr:.7f}-BS{batch_size}{prune}-Epoch{epoch}{exp_name}".format(
        dataset=args.dataset, arch=args.arch, width=args.width, bn=".BN" if args.bn else "",
        lr=args.lr, batch_size=args.batch_size, epoch=args.epochs,
        prune="-prune%.2f"%args.rand_prune if args.rand_prune > 0 else "", #  random pruning for now
        exp_name="" if args.exp_name == "" else "-"+args.exp_name) #, seed=args.seed)
    SAVE_DIR = os.path.join(args.save_dir, job_name)

    with open('all_dags_str.json') as json_file:
        all_dags_str = json.load(json_file)
    random_dag_list = np.load("random_dag_list.npy")
    if args.start_idx >= 0 and args.end_idx > 0 and args.end_idx > args.start_idx:
        random_dag_list = random_dag_list[args.start_idx:args.end_idx]
    if args.reverse_order:
        random_dag_list = random_dag_list[::-1]

    PID = os.getpid()
    print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, SAVE_DIR))

    pbar = tqdm(random_dag_list, position=0, leave=True)
    for dag_idx in pbar:
        ing_file_name = "%s"%(os.path.join(SAVE_DIR, "%d.ing"%dag_idx))
        args.save_dir = os.path.join(SAVE_DIR, "%d"%(dag_idx))
        if (os.path.isfile(ing_file_name) and abs(time.time() - check_modified_time(ing_file_name)) < ING_GAP_SEC) or ((not os.path.isfile(ing_file_name)) and os.path.exists(args.save_dir)):
            pbar.set_description("Skip DAG#%d"%dag_idx)
            continue

        args.dag = all_dags_str[dag_idx]
        if os.path.isfile(ing_file_name):
            prefix = "Resume"
        else:
            prefix = "Train"
        for r_idx in range(args.repeat):
            seed = args.seed + r_idx
            args.save_dir = os.path.join(SAVE_DIR, "%d/%d"%(dag_idx, seed))
            if (not os.path.isfile("%s/checkpoint.pth.tar"%args.save_dir)) and os.path.isfile("%s/net_train.png"%args.save_dir):
                # no ckpt but has trained: already finished
                continue
            prepare_seed(seed)
            pbar.set_description("%s DAG#%d seed %d"%(prefix, dag_idx, seed))
            train_model(dag_idx, ing_file_name)
        os.system("rm %s"%ing_file_name)


def train_model(dag_idx, ing_file_name):
    global args
    logger = prepare_logger(args, verbose=False)
    os.system("touch %s"%ing_file_name)

    if not args.inference:
        os.makedirs(args.save_dir, exist_ok=True)

    # prepare dataset
    NUM_VAL_IMAGE = 50
    c_in = 3
    if args.dataset == 'cifar10':
        classes = 10
        dummy_shape = (3, 32, 32)
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers, flatten=args.arch == "mlp")
    elif args.dataset == 'cifar100':
        classes = 100
        dummy_shape = (3, 32, 32)
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers,
                                                                     val_size=NUM_VAL_IMAGE * classes, flatten=args.arch == "mlp")
    elif args.dataset == 'svhn':
        classes = 10
        dummy_shape = (3, 32, 32)
        train_loader, val_loader, test_loader = svhn_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers, flatten=args.arch == "mlp")
    elif args.dataset == 'mnist':
        c_in = 1
        classes = 10
        dummy_shape = (1, 28, 28)
        train_loader, val_loader, test_loader = mnist_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers, flatten=args.arch == "mlp")
    elif args.dataset == 'tinyimagenet':
        classes = 200
        dummy_shape = (3, 64, 64)
        train_loader, val_loader, test_loader = imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers, flatten=args.arch == "mlp")
    elif args.dataset is None:
        pass
    else:
        raise ValueError('Dataset not supprot yet!')

    model = build_model(args, classes, dummy_shape)
    logger.log(str(model))

    # setup initialization and mask
    if args.rand_prune > 0:
        pruning_model_random_layer_specified(model, min(1, args.rand_prune)) #, conv1=args.prune_conv1)
        remain_weight_rate = check_sparsity(model)
        logger.log("remaining weight rate: %.2f"%remain_weight_rate)

    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(','))) if args.decreasing_lr else None

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = None
    if decreasing_lr:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    if args.inference:
        # test
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint)

        test_acc = validate(test_loader, model, criterion, 0)
        logger.log('* Test Accuracy = {}'.format(test_acc))
        return 0

    # if args.resume:
    ckpt_path = "%s/checkpoint.pth.tar"%args.save_dir
    if os.path.isfile(ckpt_path):
        logger.log('resume from checkpoint {}'.format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location = torch.device('cuda:'+str(args.gpu)))
        # best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = None
        if decreasing_lr:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
            scheduler.load_state_dict(checkpoint['scheduler'])

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.log('loading from epoch: ',start_epoch)#, 'best_acc=', best_acc)
    else:
        all_result = {}
        all_result['train_acc'] = []
        all_result['test_acc'] = []
        all_result['val_acc'] = []
        start_epoch = 0

    logger.log("Path {}".format(args.save_dir))
    for epoch in range(start_epoch, args.epochs):
        os.system("touch %s"%ing_file_name)

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, split="Val")
        logger.writer.add_scalar("train/loss", train_loss, epoch)
        logger.writer.add_scalar("train/accuracy", train_acc, epoch)
        logger.writer.add_scalar("validation/loss", val_loss, epoch)
        logger.writer.add_scalar("validation/accuracy", val_acc, epoch)
        if test_loader:
            test_loss, test_acc = validate(test_loader, model, criterion, epoch, split="Test")
            logger.writer.add_scalar("test/loss", test_loss, epoch)
            logger.writer.add_scalar("test/accuracy", test_acc, epoch)
            logger.log("Path {}".format(args.save_dir))
            logger.log("Epoch {} Train {:.2f} (Loss {:.4f}) Validation {:.2f} (Loss {:.4f}) Test {:.2f} (Loss {:.4f})".format(epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            all_result['test_acc'].append(test_acc)
            plt.plot(all_result['test_acc'], label='test_acc')
        else:
            logger.log("Path {}".format(args.save_dir))
            logger.log("Epoch {} Train {:.2f} (Loss {:.4f}) Validation {:.2f} (Loss {:.4f})".format(epoch, train_acc, train_loss, val_acc, val_loss))

        if scheduler: scheduler.step()

        all_result['train_acc'].append(train_acc)
        all_result['val_acc'].append(val_acc)

        checkpoint = {
            'result': all_result,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            # 'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None
        }
        save_checkpoint(checkpoint, is_best=False, save_path=args.save_dir)

        # plot training curve
        plt.plot(all_result['train_acc'], label='train_acc')
        plt.plot(all_result['val_acc'], label='val_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    #report result
    val_pick_best_epoch = np.argmax(np.array(all_result['val_acc']))
    if test_loader:
        logger.log('* best accuracy = {}, Epoch = {}'.format(all_result['test_acc'][val_pick_best_epoch], val_pick_best_epoch+1))
    else:
        logger.log('* best accuracy = {}, Epoch = {}'.format(all_result['val_acc'][val_pick_best_epoch], val_pick_best_epoch+1))
    os.system("rm %s"%ckpt_path)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            warmup_lr(args.warmup, args.lr, epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        prec1, gt_num = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], gt_num[0])

        losses.update(loss.item(), image.size(0))

    return float(losses.avg), float(top1.vec2sca_avg)


def validate(val_loader, model, criterion, epoch, split="Test"):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        prec1, gt_num = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], gt_num[0])
        losses.update(loss.item(), image.size(0))

    return float(losses.avg), float(top1.vec2sca_avg)


if __name__ == '__main__':
    main()
