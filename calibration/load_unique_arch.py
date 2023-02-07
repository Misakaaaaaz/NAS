import torch
import os
import pandas as pd
import numpy as np
from nats_bench import create
import xautodl
from xautodl.models import get_cell_based_tiny_net
from datatest import get_logits_labels, get_valid_test_loader, ECELoss, AdaptiveECELoss, ClasswiseECELoss, \
    get_logits_labels2
from temprature import ModelWithTemperature
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
'''
api = create(r"/media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)
for dset in ['cifar10', 'cifar100', 'ImageNet16-120']:
    for idx in np.random.randint(0,15625,20):
        params = api.get_net_param(idx, dset, None, hp='200')
        print(params.keys())
'''
from nats_bench import create
from nats_bench.api_utils import time_string
import numpy as np

# Create the API for topology search space
api = create(r"/media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)
unique_strs = []
for index in range(len(api)):
    unique_str = api.get_unique_str(index)
    unique_strs.append(unique_str)

print(unique_strs[:10])
#print('{:} There are {:} isomorphism architectures on the topology search space'.format(time_string(), len(set(unique_strs))))