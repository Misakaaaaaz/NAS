import numpy as np
import pandas as pd
import torch
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
from datatest import get_logits_labels, ECELoss, get_valid_test_loader, AdaptiveECELoss, ClasswiseECELoss


for dset in ['cifar100']:
#for dset in ['cifar10', 'cifar100', 'ImageNet16-120']:
    _, testloader = get_valid_test_loader(dset, '/media/linwei/disk1/NATS-Bench/cifar.python', batch=256)


    idx_list = []
    acc_list = []
    ece_list = []
    aece_list = []
    cece_list = []

    #for idx in range(15625):
    for idx in range(1500, 15625):

        api = create(r"/media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'sss',
                     fast_mode=True,
                     verbose=True)
        acc = api.get_more_info(idx, dset, hp='200', is_random=False)["test-accuracy"]
        idx_list.append(idx)
        acc_list.append(acc)
        config = api.get_net_config(idx, dset)
        network = get_cell_based_tiny_net(config)
        params = api.get_net_param(idx, dset, None, hp='200')
        network.load_state_dict(next(iter(params.values())))


        with torch.no_grad():
            net = network.cuda(0)
            logits, labels = get_logits_labels(testloader, net)
            ece_criterion = ECELoss().cuda(0)
            ece = ece_criterion(logits, labels).item()
            ece_list.append(ece)
            aece_criterion = AdaptiveECELoss().cuda(0)
            aece = aece_criterion(logits, labels).item()
            aece_list.append(aece)
            cece_criterion = ClasswiseECELoss().cuda(0)
            cece = cece_criterion(logits, labels).item()
            cece_list.append(cece)

        #print(idx_list)
        if (idx % 100 == 99) | (idx == 15624):
            df = pd.DataFrame(zip(idx_list, acc_list, ece_list, aece_list, cece_list))

            df.to_csv('/media/linwei/disk1/NATS-Bench/NATS-all/' + dset + '/to' + str(idx)+'.csv')
            #df.to_csv('/media/linwei/disk1/NATS-Bench/NATS-all-details/' + dset +'.csv')


            idx_list = []
            acc_list = []
            ece_list = []
            aece_list = []
            cece_list = []
