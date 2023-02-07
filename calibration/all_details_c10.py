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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



for dset in ['cifar10']:
    #for dset in ['ImageNet16-120']:
        valloader, testloader = get_valid_test_loader(dset, '/media/linwei/disk1/NATS-Bench/cifar.python', batch=256)

        idx_list = []
        acc_list = []
        ece_bef_list = []
        ece_aft_list = []
        aece_bef_list = []
        aece_aft_list = []
        cece_bef_list = []
        cece_aft_list = []
        #for idx in range(15625):
        #for idx in range(100):

        for idx in range(14400, 15000):
            api = create(r"/media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'tss',
                         fast_mode=True, verbose=True)
            idx_list.append(idx)
            testacc = api.get_more_info(idx, dset, hp='200', is_random=False)["test-accuracy"]
            acc_list.append(testacc)

            config = api.get_net_config(idx, dset)
            network = get_cell_based_tiny_net(config)
            params = api.get_net_param(idx, dset, None, hp='200')
            network.load_state_dict(next(iter(params.values())))

            with torch.no_grad():
                net1 = network.cuda(0)
                logits, labels = get_logits_labels(testloader, net1)
                x=[]
                y=[]
                z=[]
                for bins in [5,10,15,20]:
                    ece_criterion = ECELoss(n_bins=bins).cuda(0)
                    ece = ece_criterion(logits, labels).item()
                    x.append(ece)
                    aece_criterion = AdaptiveECELoss(n_bins=bins).cuda(0)
                    aece = aece_criterion(logits, labels).item()
                    y.append(aece)
                    cece_criterion = ClasswiseECELoss(n_bins=bins).cuda(0)
                    cece = cece_criterion(logits, labels).item()
                    z.append(cece)
                ece_bef_list.append(x)
                aece_bef_list.append(y)
                cece_bef_list.append(z)

            scaled_model = ModelWithTemperature(network)
            scaled_model.set_temperature(valloader)

            with torch.no_grad():
                net2 = scaled_model.cuda(0)
                logits, labels = get_logits_labels2(testloader, net2)
                p = []
                q = []
                r = []
                for bins in [5, 10, 15, 20]:
                    ece_criterion = ECELoss(n_bins=bins).cuda(0)
                    ece = ece_criterion(logits, labels).item()
                    p.append(ece)
                    aece_criterion = AdaptiveECELoss(n_bins=bins).cuda(0)
                    aece = aece_criterion(logits, labels).item()
                    q.append(aece)
                    cece_criterion = ClasswiseECELoss(n_bins=bins).cuda(0)
                    cece = cece_criterion(logits, labels).item()
                    r.append(cece)
                ece_aft_list.append(p)
                aece_aft_list.append(q)
                cece_aft_list.append(r)



            if (idx % 200 == 199) | (idx == 15624):

                ece_bef_list = np.array(ece_bef_list)
                ece_aft_list = np.array(ece_aft_list)
                aece_bef_list = np.array(aece_bef_list)
                aece_aft_list = np.array(aece_aft_list)
                cece_bef_list = np.array(cece_bef_list)
                cece_aft_list = np.array(cece_aft_list)
                df = pd.DataFrame(zip(idx_list, acc_list, ece_bef_list[:,0], ece_aft_list[:,0], aece_bef_list[:,0], aece_aft_list[:,0], cece_bef_list[:,0], cece_aft_list[:,0],
                                        ece_bef_list[:, 1], ece_aft_list[:, 1], aece_bef_list[:, 1], aece_aft_list[:, 1], cece_bef_list[:,1], cece_aft_list[:, 1],
                                      ece_bef_list[:, 2], ece_aft_list[:, 2], aece_bef_list[:, 2], aece_aft_list[:, 2], cece_bef_list[:,2], cece_aft_list[:, 2],
                                      ece_bef_list[:, 3], ece_aft_list[:, 3], aece_bef_list[:, 3], aece_aft_list[:, 3], cece_bef_list[:,3], cece_aft_list[:, 3],))

                idx_list = []
                acc_list = []
                ece_bef_list = []
                ece_aft_list = []
                aece_bef_list = []
                aece_aft_list = []
                cece_bef_list = []
                cece_aft_list = []
                df.to_csv('/media/linwei/disk1/NATS-Bench/NATS-details/'+dset+'/to'+str(idx)+'.csv')