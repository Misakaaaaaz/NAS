import pandas as pd
import numpy as np
import os

url = '/media/linwei/disk1/NATS-Bench/NATS-details/'

for dset in ['cifar10']:
#for dset in ['ImageNet16-120']:
    ulist = []
    file = os.listdir(url+dset)
    df = pd.DataFrame()
    for f in file:
        curl= os.path.join(url+dset,f)
        ulist.append(curl)


        #print(curl)

    df = pd.read_csv(ulist[0])
    for i in ulist[1:]:
        thisdf = pd.read_csv(i)
        df = pd.concat([df, thisdf])
    df.to_csv(url+dset+'-4binspre&post-sum.csv')


