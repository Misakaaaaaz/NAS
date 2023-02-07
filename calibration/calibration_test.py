import torch
import numpy as np
from nats_bench import create
import xautodl
from xautodl.models import get_cell_based_tiny_net
from datatest import get_logits_labels, get_test_loader, ECELoss

# Create the API instance for the size search space in NATS
api = create(r"X:/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)
#api = create(r"X:/NATS-Bench/NATS-sss-v1_0-50262-/NATS-sss-v1_0-50262-full", 'sss', fast_mode=True, verbose=True)
#api = create(r"X:/NATS-Bench/NATS-sss-v1_0-50262-simple", 'sss', fast_mode=True, verbose=True)
config = api.get_net_config(12, 'cifar10')
network = get_cell_based_tiny_net(config)
params = api.get_net_param(12, 'cifar10', None)
network.load_state_dict(next(iter(params.values())))

test_loader = get_test_loader(
            batch_size=128,
            pin_memory=True)

net = network.cuda(0)
logits, labels = get_logits_labels(test_loader, net)
ece_criterion = ECELoss().cuda(0)
ece = ece_criterion(logits, labels).item()
print("ece = {:.4f}".format(ece))