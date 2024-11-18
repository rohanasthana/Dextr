'''
    collect manifold complexity
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
from zmq import CURVE
from thop import profile

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from length import curve_complexity_differentiable, get_extrinsic_curvature, get_curve_input

# from models import model_dict
import models
from dag_utils import effective_depth_width, dag2affinity, find_all_dags, build_model
from dataset import cifar10_dataloaders, cifar100_dataloaders, svhn_dataloaders, mnist_dataloaders, imagenet_dataloaders
from logger import prepare_seed, prepare_logger
from utils import save_checkpoint, warmup_lr, AverageMeter, accuracy

from pdb import set_trace as bp

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training Subnetworks')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='mlp', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--dag', default=None, help='from-to edges separated by underscore. 0: broken edge; 1: skip-connect; 2: linear or conv')
parser.add_argument('--imagenet_arch', action="store_true", help="back to imagenet architecture (conv1, maxpool)")
parser.add_argument('--width', type=int, default=2048, help='hidden width')
parser.add_argument('--bn', action="store_true", help="use BN")
parser.add_argument('--fc', action="store_true", help="use FC in ConvNet")

##################################### General setting ############################################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='./experiment', type=str)
parser.add_argument('--exp_name', help='additional names for experiment', default='', type=str)

##################################### Experiment setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--repeat', type=int, default=3, help='repeat calculation')


best_acc = 0
args = parser.parse_args()

if args.seed is None:
    args.seed = random.randint(0, 999)
prepare_seed(args.seed)

PID = os.getpid()
print("<< ============== PID = %d ============== >>"%(PID))


def main():
    global args, best_acc

    job_name = "TRAVERSAL-LENGTH-{dataset}-{arch}{width}{bn}{dag}-BS{batch_size}{exp_name}".format(
        dataset=args.dataset, arch=args.arch, width=args.width, bn=".BN" if args.bn else "",
        dag="_"+args.dag if args.dag else "", batch_size=args.batch_size,
        exp_name="" if args.exp_name == "" else "-"+args.exp_name)
    timestamp = "{:}".format(time.strftime("%m%d%H%M%S", time.gmtime()))
    args.save_dir = os.path.join(args.save_dir, job_name, "seed%d_"%args.seed+timestamp)
    logger = prepare_logger(args)

    torch.cuda.set_device(int(args.gpu))
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

    OP_SPACE = [0, 1, 2]
    ALL_DAGS = find_all_dags([], max_num_nodes=4, candidate_ops=OP_SPACE)
    ALL_DAGS_valid = []
    LEs_all = []
    kappas_all = []
    Xs, _ = next(iter(train_loader))

    THETA, CURVE_INPUT = get_curve_input((128, *dummy_shape))

    for dag in tqdm(ALL_DAGS):
        depth, width, depth_total = effective_depth_width(dag2affinity(dag))
        # ALL_DAGS_valid.append([dag, depth, width, sum([np.where(np.array(dag[_node]) == 2)[0].shape[0] for _node in range(3)])]) # #parameterized edges
        ALL_DAGS_valid.append([dag, depth, width, depth_total]) # #parameterized edges
        if depth == 0:
            LEs_all.append([-1] * args.repeat)
            kappas_all.append([-1] * args.repeat)
            continue

        LEs_all.append([])
        kappas_all.append([])
        args.dag = '_'.join([''.join([str(edge) for edge in node]) for node in dag])
        model = build_model(args, classes, dummy_shape)
        macs, params = profile(model, inputs=(torch.randn(1, *dummy_shape), ), verbose=False)
        model = model.cuda()
        for _ in range(args.repeat):
            model._init()
            _le = curve_complexity_differentiable(model, curve_inputs=(THETA, CURVE_INPUT), batch_size=128, train_mode=True, need_graph=True, reduction='mean')
            LEs_all[-1].append(_le.cpu().detach().numpy().tolist())
            _kappa = get_extrinsic_curvature(model, curve_inputs=(THETA, CURVE_INPUT), batch_size=128, train_mode=True)
            kappas_all[-1].append(_kappa)

    np.save(os.path.join(args.save_dir, "dags_valid.npy"), ALL_DAGS_valid)
    np.save(os.path.join(args.save_dir, "LEs_all.npy"), LEs_all)
    np.save(os.path.join(args.save_dir, "kappas_all.npy"), kappas_all)


if __name__ == '__main__':
    main()
