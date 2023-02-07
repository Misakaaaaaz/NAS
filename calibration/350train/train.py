from torch import optim
import torch
import torch.utils.data
import argparse
import torch.backends.cudnn as cudnn
import random
import json
import sys
import os
import os

import numpy as np
import pandas as pd
import math
import torch
import numpy as np
from nats_bench import create
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
import xautodl
from xautodl.datasets.DownsampledImageNet import ImageNet16
from xautodl.datasets.SearchDatasetWrap import SearchDataset
from xautodl.config_utils import load_config
import torch
import torchvision.models as M
from torch.optim.lr_scheduler import CosineAnnealingLR
from xautodl.models import get_cell_based_tiny_net

from datatest import get_valid_test_loader, ECELoss, AdaptiveECELoss, ClasswiseECELoss, get_logits_labels
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn, optim
from torch.nn import functional as F


torch.manual_seed(777)
url='/media/linwei/disk1/NATS-Bench/350cifar10/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dset='cifar10'

cuda = False
if (torch.cuda.is_available() and args.gpu):
    cuda = True
device = torch.device("cuda" if cuda else "cpu")
print("CUDA set: " + str(cuda))

num_classes = dataset_num_classes[args.dataset]

# Choosing the model to train
net = models[args.model](num_classes=num_classes)

# Setting model name
if args.model_name is None:
    args.model_name = args.model

if args.gpu is True:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

start_epoch = 0
num_epochs = args.epoch
if args.load:
    net.load_state_dict(torch.load(args.save_loc + args.saved_model_name))
    start_epoch = int(args.saved_model_name[args.saved_model_name.rfind('_') + 1:args.saved_model_name.rfind('.model')])

if args.optimiser == "sgd":
    opt_params = net.parameters()
    optimizer = optim.SGD(opt_params,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
elif args.optimiser == "adam":
    opt_params = net.parameters()
    optimizer = optim.Adam(opt_params,
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone],
                                           gamma=0.1)

if (args.dataset == 'tiny_imagenet'):
    train_loader = dataset_loader[args.dataset].get_data_loader(
        root=args.dataset_root,
        split='train',
        batch_size=args.train_batch_size,
        pin_memory=args.gpu)

    val_loader = dataset_loader[args.dataset].get_data_loader(
        root=args.dataset_root,
        split='val',
        batch_size=args.test_batch_size,
        pin_memory=args.gpu)

    test_loader = dataset_loader[args.dataset].get_data_loader(
        root=args.dataset_root,
        split='val',
        batch_size=args.test_batch_size,
        pin_memory=args.gpu)
else:
    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        random_seed=1,
        pin_memory=args.gpu
    )

    test_loader = dataset_loader[args.dataset].get_test_loader(
        batch_size=args.test_batch_size,
        pin_memory=args.gpu
    )

training_set_loss = {}
val_set_loss = {}
test_set_loss = {}
val_set_err = {}

for epoch in range(0, start_epoch):
    scheduler.step()

best_val_acc = 0
for epoch in range(start_epoch, num_epochs):
    scheduler.step()
    if (args.loss_function == 'focal_loss' and args.gamma_schedule == 1):
        if (epoch < args.gamma_schedule_step1):
            gamma = args.gamma
        elif (epoch >= args.gamma_schedule_step1 and epoch < args.gamma_schedule_step2):
            gamma = args.gamma2
        else:
            gamma = args.gamma3
    else:
        gamma = args.gamma

    train_loss = train_single_epoch(epoch,
                                    net,
                                    train_loader,
                                    optimizer,
                                    device,
                                    loss_function=args.loss_function,
                                    gamma=gamma,
                                    lamda=args.lamda,
                                    loss_mean=args.loss_mean)
    val_loss = test_single_epoch(epoch,
                                 net,
                                 val_loader,
                                 device,
                                 loss_function=args.loss_function,
                                 gamma=gamma,
                                 lamda=args.lamda)
    test_loss = test_single_epoch(epoch,
                                  net,
                                  val_loader,
                                  device,
                                  loss_function=args.loss_function,
                                  gamma=gamma,
                                  lamda=args.lamda)
    _, val_acc, _, _, _ = test_classification_net(net, val_loader, device)

    training_set_loss[epoch] = train_loss
    val_set_loss[epoch] = val_loss
    test_set_loss[epoch] = test_loss
    val_set_err[epoch] = 1 - val_acc

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print('New best error: %.4f' % (1 - best_val_acc))
        save_name = args.save_loc + \
                    args.model_name + '_' + \
                    loss_function_save_name(args.loss_function, args.gamma_schedule, gamma, args.gamma, args.gamma2,
                                            args.gamma3, args.lamda) + \
                    '_best_' + \
                    str(epoch + 1) + '.model'
        torch.save(net.state_dict(), save_name)

    if (epoch + 1) % args.save_interval == 0:
        save_name = args.save_loc + \
                    args.model_name + '_' + \
                    loss_function_save_name(args.loss_function, args.gamma_schedule, gamma, args.gamma, args.gamma2,
                                            args.gamma3, args.lamda) + \
                    '_' + str(epoch + 1) + '.model'
        torch.save(net.state_dict(), save_name)
