import os

import numpy as np
import pandas as pd
import math
import torch
import numpy as np
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

from datatest import get_valid_test_loader, ECELoss, AdaptiveECELoss, ClasswiseECELoss
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn, optim
from torch.nn import functional as F
#cifar10>935
#a1 = [1, 42, 69, 81, 99, 111, 129, 138, 168, 203, 244, 250, 258, 265, 295, 322, 352, 378, 439, 541, 543, 571, 582, 587, 654, 655, 684, 690, 699, 722, 844, 857, 869, 873, 937, 1000, 1027, 1062, 1114, 1177, 1317, 1414, 1425, 1459, 1462, 1556, 1583, 1617, 1654, 1656, 1669, 1716, 1842, 1856, 1906, 1941, 1963, 1965, 2013, 2018, 2030, 2053, 2066, 2159, 2162, 2210, 2225, 2246, 2254, 2472, 2508, 2510, 2528, 2538, 2566, 2580, 2581, 2676, 2704, 2725, 2772, 2776, 2778, 3035, 3045, 3166, 3255, 3383, 3385, 3456, 3540, 3558, 3621, 3625, 3731, 3803, 3888, 3907, 3927, 3928, 3982, 3986, 4062, 4099, 4141, 4181, 4208, 4237, 4244, 4361, 4385, 4418, 4511, 4533, 4557, 4562, 4641, 4681, 4747, 4800, 4902, 4907, 4926, 4946, 4954, 4983, 5087, 5095, 5111, 5115, 5176, 5256, 5292, 5369, 5374, 5525, 5539, 5544, 5664, 5698, 5711, 5721, 5731, 5749, 5776, 5780, 5801, 5823, 5883, 5902, 5979, 6016, 6022, 6111, 6118, 6119, 6304, 6330, 6489, 6499, 6515, 6530, 6584, 6640, 6676, 6697, 6747, 6804, 6830, 6931, 6993, 6999, 7209, 7313, 7353, 7384, 7407, 7419, 7463, 7482, 7548, 7556, 7570, 7602, 7619, 7632, 7725, 7897, 7906, 7963, 8002, 8058, 8114, 8135, 8159, 8162, 8216, 8253, 8285, 8288, 8290, 8401, 8413, 8426, 8449, 8452, 8529, 8572, 8592, 8606, 8610, 8621, 8649, 8654, 8695, 8708, 8733, 8736, 8845, 8855, 8860, 8877, 8884, 9016, 9046, 9099, 9121, 9149, 9161, 9302, 9431, 9493, 9572, 9586, 9612, 9640, 9648, 9656, 9718, 9722, 9743, 9819, 9916, 9923, 9930, 9941, 9950, 9972, 9997, 10000, 10007, 10063, 10073, 10086, 10131, 10154, 10165, 10188, 10189, 10192, 10200, 10279, 10335, 10337, 10360, 10370, 10439, 10476, 10484, 10545, 10603, 10676, 10699, 10715, 10721, 10733, 10749, 10750, 10767, 10835, 10838, 10872, 10902, 10935, 10948, 10975, 11145, 11146, 11157, 11174, 11217, 11218, 11226, 11232, 11270, 11271, 11298, 11299, 11307, 11351, 11372, 11376, 11400, 11443, 11465, 11472, 11564, 11584, 11692, 11711, 11722, 11737, 11754, 11777, 11796, 11797, 11876, 12002, 12043, 12123, 12158, 12295, 12384, 12399, 12415, 12421, 12451, 12464, 12466, 12476, 12491, 12574, 12576, 12606, 12608, 12617, 12645, 12663, 12687, 12692, 12761, 12799, 12839, 12956, 12967, 13032, 13112, 13127, 13166, 13171, 13325, 13363, 13376, 13390, 13402, 13417, 13451, 13539, 13551, 13585, 13644, 13652, 13714, 13755, 13778, 13801, 13809, 13869, 13925, 13981, 13987, 13991, 14029, 14118, 14139, 14152, 14174, 14203, 14029, 14118, 14139, 14152, 14174, 14203, 14218, 14228, 14296, 14375, 14443, 14462, 14469, 14572, 14580, 14638, 14654, 14669, 14709, 14714, 14720, 14805, 14861, 14866, 14985, 15028, 15144, 15159, 15283, 15341, 15383, 15425, 15461, 15485, 15508, 15537, 15567, 15586, 15615]
#cifar100>71
#a2 = [69, 81, 168, 265, 543, 582, 655, 690, 699, 722, 857, 862, 869, 937, 1317, 1346, 1425, 1459, 1462, 1556, 1583, 1656, 2018, 2030, 2053, 2159, 2210, 2508, 2538, 2554, 2566, 2580, 2581, 2725, 2772, 2778, 2889, 3035, 3192, 3383, 3385, 3456, 3540, 3621, 3731, 3888, 3928, 3982, 4062, 4141, 4208, 4237, 4244, 4361, 4418, 4557, 4562, 4747, 4907, 4926, 4954, 5111, 5115, 5207, 5292, 5374, 5711, 5749, 5979, 6111, 6118, 6119, 6304, 6530, 6830, 6925, 6931, 7209, 7294, 7337, 7419, 7548, 7556, 7570, 7602, 7725, 7897, 7906, 8002, 8004, 8102, 8114, 8135, 8162, 8216, 8290, 8449, 8452, 8529, 8572, 8610, 8621, 8625, 8695, 8768, 8853, 8860, 9149, 9431, 9586, 9635, 9648, 9718, 9930, 9972, 10000, 10007, 10154, 10165, 10189, 10315, 10321, 10337, 10360, 10400, 10436, 10439, 10476, 10603, 10676, 10699, 10715, 10721, 10749, 10770, 10835, 10902, 10948, 10957, 11000, 11298, 11307, 11400, 11465, 11584, 11692, 11711, 11722, 11754, 11797, 11876, 11915, 12076, 12201, 12295, 12384, 12399, 12415, 12421, 12574, 12576, 12687, 12761, 12799, 12887, 12919, 12967, 13363, 13390, 13403, 13417, 13539, 13551, 13644, 13714, 13778, 13925, 13981, 13991, 14174, 14296, 14375, 14443, 14462, 14572, 14624, 14654, 14714, 14805, 14861, 14866, 15190, 15383, 15425, 15485, 15510]
#ImageNet>45
#a3 = [65, 81, 109, 129, 168, 203, 265, 352, 374, 435, 439, 542, 571, 582, 648, 655, 699, 857, 862, 869, 915, 1013, 1027, 1084, 1237, 1317, 1414, 1425, 1459, 1555, 1556, 1716, 1742, 1939, 2018, 2030, 2053, 2159, 2162, 2210, 2212, 2213, 2217, 2246, 2272, 2387, 2508, 2538, 2554, 2581, 2700, 2720, 2743, 2778, 2807, 2889, 2934, 3192, 3275, 3385, 3424, 3456, 3458, 3540, 3558, 3580, 3621, 3697, 3731, 3738, 3771, 3818, 3888, 3928, 3934, 3982, 4013, 4062, 4133, 4135, 4208, 4212, 4237, 4244, 4275, 4361, 4418, 4424, 4496, 4557, 4562, 4681, 4707, 4747, 4902, 4926, 4954, 4986, 5063, 5111, 5176, 5292, 5301, 5374, 5584, 5633, 5711, 5721, 5749, 6111, 6118, 6225, 6231, 6304, 6375, 6459, 6530, 6585, 6640, 6676, 6687, 6932, 6982, 7129, 7176, 7293, 7337, 7408, 7430, 7442, 7463, 7548, 7602, 7826, 7855, 7901, 7906, 7926, 8002, 8058, 8102, 8114, 8135, 8162, 8290, 8449, 8452, 8572, 8621, 8695, 8733, 8736, 8768, 8827, 8853, 8860, 9080, 9106, 9149, 9274, 9372, 9493, 9586, 9596, 9608, 9836, 9902, 9903, 9923, 9930, 9972, 10000, 10007, 10066, 10087, 10154, 10165, 10189, 10315, 10324, 10337, 10360, 10400, 10436, 10439, 10593, 10603, 10676, 10699, 10715, 10721, 10733, 10749, 10793, 10838, 10902, 10935, 10948, 11060, 11145, 11298, 11307, 11400, 11564, 11584, 11711, 11777, 11797, 11876, 11882, 11887, 11915, 12015, 12068, 12080, 12123, 12156, 12201, 12384, 12411, 12415, 12466, 12476, 12572, 12574, 12606, 12677, 12687, 12702, 12724, 12761, 12887, 12922, 12967, 13111, 13127, 13171, 13390, 13417, 13532, 13583, 13644, 13714, 13717, 13778, 13814, 13904, 13925, 14016, 14029, 14039, 14118, 14140, 14174, 14217, 14375, 14417, 14443, 14538, 14640, 14720, 14840, 14861, 14866, 14872, 14985, 15089, 15159, 15166, 15190, 15257, 15383, 15446, 15508, 15510, 15524, 15559, 15599]

def get_train_loader(dataset, config_root, batch):
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

        cifar_split = load_config("{:}/cifar-split.txt".format(config_root), None, None)
        train_split, valid_split = (
            cifar_split.train,
            cifar_split.valid,
        )
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
            num_workers=1,
            pin_memory=True
        )
    return train_loader

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


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dset='cifar10'
#trainloader = get_train_loader(dset, '/media/linwei/disk1/NATS-Bench/cifar.python', batch=256)
valloader, testloader = get_valid_test_loader(dset, '/media/linwei/disk1/NATS-Bench/cifar.python', batch=256)

#model = M.resnet50(pretrained=True)

model = M.resnet50()
dim_in = model.fc.in_features
model.fc =nn.Linear(in_features=dim_in, out_features=10, bias=True)
model = model.cuda(0)

epochs=200
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9, nesterov=True)
schedular = CosineAnnealingLR(optimizer, T_max=epochs)
lossfunction =nn.CrossEntropyLoss()



for epoch in range(epochs):
    trainloader = get_train_loader(dset, '/media/linwei/disk1/NATS-Bench/cifar.python', batch=256)
    model.train()
    for input, label in trainloader:
        input = input.cuda(0)
        label = label.cuda(0)
        logits = model(input)
        softmaxes = F.softmax(logits, dim=1)
        #print(logits)
        #loss = lossfunction(logits, label)
        loss = lossfunction(softmaxes, label)
        #print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    schedular.step()

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
model.eval()
with torch.no_grad():
    info=[]
    logits, labels = get_logits_labels2(testloader, model)
    softmaxes = F.softmax(logits, dim=1)
    _, predictions = torch.max(softmaxes, 1)
    acc= predictions.eq(labels).float().mean()
    info.append(acc)
    ece_criterion = ECELoss().cuda(0)
    ece = ece_criterion(logits, labels).item()

    info.append(ece)

print(info)
''''''

#print(model)

#print(a1[200])
#x=[[1,2],[3,4]]
#arr = np.array(x)
#y = [1,2]
#df = pd.DataFrame(zip(arr[:,0],arr[:,1],y))
#print([x for x in range(25)])
#'ImageNet16-120'
#val, test = get_valid_test_loader('ImageNet16-120', '/media/linwei/disk1/NATS-Bench/cifar.python', batch=256)

'''
nli=[1,2,3,4,5]

nl = np.array(nli)

print(np.mean(nl), np.std(nl))
'''


#a = [1, 42, 69, 81, 99, 111, 129, 138, 168, 203, 244, 250, 258, 265, 295, 322, 352, 378, 439, 541, 543, 571, 582, 587, 654, 655, 684, 690, 699, 722, 844, 857, 869, 873, 937, 1000, 1027, 1062, 1114, 1177, 1317, 1414, 1425, 1459, 1462, 1556, 1583, 1617, 1654, 1656, 1669, 1716, 1842, 1856, 1906, 1941, 1963, 1965, 2013, 2018, 2030, 2053, 2066, 2159, 2162, 2210, 2225, 2246, 2254, 2472, 2508, 2510, 2528, 2538, 2566, 2580, 2581, 2676, 2704, 2725, 2772, 2776, 2778, 3035, 3045, 3166, 3255, 3383, 3385, 3456, 3540, 3558, 3621, 3625, 3731, 3803, 3888, 3907, 3927, 3928, 3982, 3986, 4062, 4099, 4141, 4181, 4208, 4237, 4244, 4361, 4385, 4418, 4511, 4533, 4557, 4562, 4641, 4681, 4747, 4800, 4902, 4907, 4926, 4946, 4954, 4983, 5087, 5095, 5111, 5115, 5176, 5256, 5292, 5369, 5374, 5525, 5539, 5544, 5664, 5698, 5711, 5721, 5731, 5749, 5776, 5780, 5801, 5823, 5883, 5902, 5979, 6016, 6022, 6111, 6118, 6119, 6304, 6330, 6489, 6499, 6515, 6530, 6584, 6640, 6676, 6697, 6747, 6804, 6830, 6931, 6993, 6999, 7209, 7313, 7353, 7384, 7407, 7419, 7463, 7482, 7548, 7556, 7570, 7602, 7619, 7632, 7725, 7897, 7906, 7963, 8002, 8058, 8114, 8135, 8159, 8162, 8216, 8253, 8285, 8288, 8290, 8401, 8413, 8426, 8449, 8452, 8529, 8572, 8592, 8606, 8610, 8621, 8649, 8654, 8695, 8708, 8733, 8736, 8845, 8855, 8860, 8877, 8884, 9016, 9046, 9099, 9121, 9149, 9161, 9302, 9431, 9493, 9572, 9586, 9612, 9640, 9648, 9656, 9718, 9722, 9743, 9819, 9916, 9923, 9930, 9941, 9950, 9972, 9997, 10000, 10007, 10063, 10073, 10086, 10131, 10154, 10165, 10188, 10189, 10192, 10200, 10279, 10335, 10337, 10360, 10370, 10439, 10476, 10484, 10545, 10603, 10676, 10699, 10715, 10721, 10733, 10749, 10750, 10767, 10835, 10838, 10872, 10902, 10935, 10948, 10975, 11145, 11146, 11157, 11174, 11217, 11218, 11226, 11232, 11270, 11271, 11298, 11299, 11307, 11351, 11372, 11376, 11400, 11443, 11465, 11472, 11564, 11584, 11692, 11711, 11722, 11737, 11754, 11777, 11796, 11797, 11876, 12002, 12043, 12123, 12158, 12295, 12384, 12399, 12415, 12421, 12451, 12464, 12466, 12476, 12491, 12574, 12576, 12606, 12608, 12617, 12645, 12663, 12687, 12692, 12761, 12799, 12839, 12956, 12967, 13032, 13112, 13127, 13166, 13171, 13325, 13363, 13376, 13390, 13402, 13417, 13451, 13539, 13551, 13585, 13644, 13652, 13714, 13755, 13778, 13801, 13809, 13869, 13925, 13981, 13987, 13991, 14029, 14118, 14139, 14152, 14174, 14203, 14029, 14118, 14139, 14152, 14174, 14203, 14218, 14228, 14296, 14375, 14443, 14462, 14469, 14572, 14580, 14638, 14654, 14669, 14709, 14714, 14720, 14805, 14861, 14866, 14985, 15028, 15144, 15159, 15283, 15341, 15383, 15425, 15461, 15485, 15508, 15537, 15567, 15586, 15615]
#a = [65, 81, 109, 129, 168, 203, 265, 352, 374, 435, 439, 542, 571, 582, 648, 655, 699, 857, 862, 869, 915, 1013, 1027, 1084, 1237, 1317, 1414, 1425, 1459, 1555, 1556, 1716, 1742, 1939, 2018, 2030, 2053, 2159, 2162, 2210, 2212, 2213, 2217, 2246, 2272, 2387, 2508, 2538, 2554, 2581, 2700, 2720, 2743, 2778, 2807, 2889, 2934, 3192, 3275, 3385, 3424, 3456, 3458, 3540, 3558, 3580, 3621, 3697, 3731, 3738, 3771, 3818, 3888, 3928, 3934, 3982, 4013, 4062, 4133, 4135, 4208, 4212, 4237, 4244, 4275, 4361, 4418, 4424, 4496, 4557, 4562, 4681, 4707, 4747, 4902, 4926, 4954, 4986, 5063, 5111, 5176, 5292, 5301, 5374, 5584, 5633, 5711, 5721, 5749, 6111, 6118, 6225, 6231, 6304, 6375, 6459, 6530, 6585, 6640, 6676, 6687, 6932, 6982, 7129, 7176, 7293, 7337, 7408, 7430, 7442, 7463, 7548, 7602, 7826, 7855, 7901, 7906, 7926, 8002, 8058, 8102, 8114, 8135, 8162, 8290, 8449, 8452, 8572, 8621, 8695, 8733, 8736, 8768, 8827, 8853, 8860, 9080, 9106, 9149, 9274, 9372, 9493, 9586, 9596, 9608, 9836, 9902, 9903, 9923, 9930, 9972, 10000, 10007, 10066, 10087, 10154, 10165, 10189, 10315, 10324, 10337, 10360, 10400, 10436, 10439, 10593, 10603, 10676, 10699, 10715, 10721, 10733, 10749, 10793, 10838, 10902, 10935, 10948, 11060, 11145, 11298, 11307, 11400, 11564, 11584, 11711, 11777, 11797, 11876, 11882, 11887, 11915, 12015, 12068, 12080, 12123, 12156, 12201, 12384, 12411, 12415, 12466, 12476, 12572, 12574, 12606, 12677, 12687, 12702, 12724, 12761, 12887, 12922, 12967, 13111, 13127, 13171, 13390, 13417, 13532, 13583, 13644, 13714, 13717, 13778, 13814, 13904, 13925, 14016, 14029, 14039, 14118, 14140, 14174, 14217, 14375, 14417, 14443, 14538, 14640, 14720, 14840, 14861, 14866, 14872, 14985, 15089, 15159, 15166, 15190, 15257, 15383, 15446, 15508, 15510, 15524, 15559, 15599]

#print(torch.cuda.memory_summary())

#nli=[1,2,3,4,5]

#df = pd.DataFrame(nli)
#df.to_csv('/media/linwei/disk1/NATS-Bench/test.csv')

#df1 = pd.read_csv('/media/linwei/disk1/NATS-Bench/test.csv', usecols=[0])
#ls = np.array(df1[1])
#print(len(a))
