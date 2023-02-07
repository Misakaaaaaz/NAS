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
from torch.optim import Optimizer

'''
class _LRScheduler(object):
    def __init__(self, optimizer, warmup_epochs, epochs):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{:} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.max_epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.current_iter = 0

    def extra_repr(self):
        return ""

    def __repr__(self):
        return "{name}(warmup={warmup_epochs}, max-epoch={max_epochs}, current::epoch={current_epoch}, iter={current_iter:.2f}".format(
            name=self.__class__.__name__, **self.__dict__
        ) + ", {:})".format(
            self.extra_repr()
        )

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def get_min_info(self):
        lrs = self.get_lr()
        return "#LR=[{:.6f}~{:.6f}] epoch={:03d}, iter={:4.2f}#".format(
            min(lrs), max(lrs), self.current_epoch, self.current_iter
        )

    def get_min_lr(self):
        return min(self.get_lr())

    def update(self, cur_epoch, cur_iter):
        if cur_epoch is not None:
            assert (
                isinstance(cur_epoch, int) and cur_epoch >= 0
            ), "invalid cur-epoch : {:}".format(cur_epoch)
            self.current_epoch = cur_epoch
        if cur_iter is not None:
            assert (
                isinstance(cur_iter, float) and cur_iter >= 0
            ), "invalid cur-iter : {:}".format(cur_iter)
            self.current_iter = cur_iter
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, T_max, eta_min):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, warmup_epochs, epochs)

    def extra_repr(self):
        return "type={:}, T-max={:}, eta-min={:}".format(
            "cosine", self.T_max, self.eta_min
        )

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if (
                self.current_epoch >= self.warmup_epochs
                and self.current_epoch < self.max_epochs
            ):
                last_epoch = self.current_epoch - self.warmup_epochs
                # if last_epoch < self.T_max:
                # if last_epoch < self.max_epochs:
                lr = (
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * last_epoch / self.T_max))
                    / 2
                )
                # else:
                #  lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.T_max-1.0) / self.T_max)) / 2
            elif self.current_epoch >= self.max_epochs:
                lr = self.eta_min
            else:
                lr = (
                    self.current_epoch / self.warmup_epochs
                    + self.current_iter / self.warmup_epochs
                ) * base_lr
            lrs.append(lr)
        return lrs
'''
def get_train_loader(dataset,  batch):
    data_dir = '/media/linwei/disk1/NATS-Bench/cifar.python'

    if dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[x / 255 for x in [125.3, 123.0, 113.9]],
            std=[x / 255 for x in [63.0, 62.1, 66.7]],
        )

        # define transform
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        data = datasets.CIFAR10(
            root=data_dir, train=True,
            download=False, transform=transform,
        )

        num_train = len(data)
        print("num_train:{}".format(num_train))
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))
        #split = 0

        np.random.seed(777)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            data, batch_size=batch, sampler=train_sampler,
            num_workers=1, pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            data, batch_size=batch, sampler=valid_sampler,
            num_workers=1, pin_memory=True,
        )


    return train_loader, valid_loader

def get_test_loader(batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[x / 255 for x in [125.3, 123.0, 113.9]],
        std=[x / 255 for x in [63.0, 62.1, 66.7]],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = '/media/linwei/disk1/NATS-Bench/cifar.python'
    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=False, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_logits_labels2(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    #with torch.no_grad():
    for data, label in data_loader:
        data = data.cuda(0)
        #data = data.cpu()

        #
        #print(data)
        #print(data.shape)
        logits = net(data)
        #print(logits)
        #print(logits[1].shape)
        logits_list.append(logits)
        labels_list.append(label)
        '''
        logits = torch.cat(logits_list).cuda(0)
        labels = torch.cat(labels_list).cuda(0)
        '''
        logits = torch.cat(logits_list).cpu()
        labels = torch.cat(labels_list).cpu()
    return logits, labels

url='/media/linwei/disk1/NATS-Bench/350cifar10/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dset='cifar10'
#trainloader = get_train_loader(dset, '/media/linwei/disk1/NATS-Bench/cifar.python', batch=256)
testloader = get_test_loader(batch_size=128)
trainloader, valloader = get_train_loader(dset,  batch=128)
#model = M.resnet50(pretrained=True)
'''
model = M.resnet50()
dim_in = model.fc.in_features
model.fc =nn.Linear(in_features=dim_in, out_features=10, bias=True)
model = model.cuda(0)
'''
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
torch.manual_seed(777)
torch.cuda.manual_seed(777)
starter.record()
idx=6111
api = create(r"/media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)
config = api.get_net_config(idx, dset)
model = get_cell_based_tiny_net(config)
model = model.cuda(0)
epochs=350
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9, nesterov=True)
#try1=100,200,300
#try2=200,250,300
#try4:100,150,200
#try5:100,150,200,250,300,350
#try5:100,150,200,250,300
#try6:1000,200,300
leng=len(trainloader)
warmups=20*len(trainloader)
def lambdastep(cur):
    if cur<warmups :
        return cur / warmups

    elif cur < 100*leng:
        return 1
    elif cur < 200 * leng:
        return 0.1
    elif cur < 300 * leng:
        return 0.05
    else:
        return 0.01




#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 175, 250], gamma=0.1)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdastep)
#scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
#scheduler = CosineAnnealingLR(optimizer, T_max=epochs*len(trainloader))
#lossfunction =nn.CrossEntropyLoss()
lossfunction =nn.CrossEntropyLoss(reduction='sum')


best_val_acc = 0
best_model_dict={}
ep_num=0
val_acc=[]
test_acc=[]
ep_list=[]
print(epochs)
for epoch in range(epochs):

    print(epoch)
    ep_list.append(epoch)
    for i, (input, label) in enumerate(trainloader):
        model.train()
        #scheduler.update(None, 1.0 * i / len(trainloader))
        input = input.cuda(0)
        label = label.cuda(0)
        optimizer.zero_grad()


        logits = model(input)
        #softmaxes = F.softmax(logits[1], dim=1)
        #print(logits)
        #loss = lossfunction(logits, label)
        loss = lossfunction(logits[1], label)
        #print(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        scheduler.step()


    model.eval()
    with torch.no_grad():
        logits, labels = get_logits_labels(valloader, model)
        softmaxes = F.softmax(logits, dim=1)
        _, predictions = torch.max(softmaxes, 1)
        acc = predictions.eq(labels).float().mean().item()
            #print(acc)
        if acc > best_val_acc:
            best_model_dict = model.state_dict()
            ep_num=epoch
            best_val_acc = acc
        val_acc.append(acc)
        print("Val_acc:{}".format(acc))


        logits, labels = get_logits_labels(testloader, model)
        softmaxes = F.softmax(logits, dim=1)
        _, predictions = torch.max(softmaxes, 1)
        acc = predictions.eq(labels).float().mean().item()
        test_acc.append(acc)
        print("Test_acc:{}".format(acc))



    #scheduler.step()
    #if epoch % 10 == 9:

    '''
    if epoch%10==9:
        print(epoch)
        model.eval()
        with torch.no_grad():
            logits, labels = get_logits_labels2(valloader, model)
            softmaxes = F.softmax(logits, dim=1)
            #_, predictions = torch.max(logits, 1)
            _, predictions = torch.max(softmaxes, 1)
            acc = predictions.eq(labels).float().mean().item()
            print(acc)
    '''
trys='try6'
modelurl = url+'idx'+str(idx)+'+epoch'+str(ep_num)+'-350-step-yy-'+trys+'.model'
df = pd.DataFrame(zip(ep_list, val_acc, test_acc))

df.to_csv(url+str(idx)+'350-step-yy-'+trys+'.csv')
torch.save(best_model_dict, modelurl)
ender.record()


model1 = get_cell_based_tiny_net(config)
model1.load_state_dict(torch.load(modelurl))
model1 = model1.cuda(0)
model1.eval()
with torch.no_grad():
    info=[]
    logits, labels = get_logits_labels(testloader, model1)
    softmaxes = F.softmax(logits, dim=1)
    _, predictions = torch.max(softmaxes, 1)
    nacc = predictions.eq(labels).float().mean()
    info.append(nacc)
    ece_criterion = ECELoss().cuda(0)
    nece = ece_criterion(logits, labels).item()

    info.append(nece)


print(starter.elapsed_time(ender))
print(info)

