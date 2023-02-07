#Qualitative comparison of test-set accuracy and PGD accuracy of NAS
# and hand-crafted architectures on CIFAR-10 dataset.
#Bubble size represents the number of parameters.

#hand
#Wide-ResNet，Inception, ResNet50/18/100, VGG, DenseNet,


#NATS, acc top5, ece top5?


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#df = pd.read_excel('/media/linwei/disk1/NATS-Bench/cifar10-gt935-sum.xlsx', usecols=[0,1,2,8])
#df = pd.read_excel('/media/linwei/disk1/NATS-Bench/imagenet16120-gt45-sum.xlsx', usecols=[0,1,2,8])
#print(df['test_acc'].tolist())
df = pd.read_csv('/media/linwei/disk1/NATS-Bench/NATS-details/cifar10-4binspre&post-sum.csv', usecols=[2,3,4])
#df = pd.read_csv('/media/linwei/disk1/NATS-Bench/NATS-details/ImageNet16-120-4binspre&post-sum.csv', usecols=[2,3,4])

print(df.keys())
#cifar10 94.125
#cifar100 72.64666668
#imgnet 46.44999988


#ece
x=df['2'].tolist()
x2=df[df['1']>=94.125]['2'].tolist()
#testacc
y=df['1'].tolist()
y2=df[df['1']>=94.125]['1'].tolist()

#params
#area=np.array(df['params'].tolist())*25
#colors random?
#colors=np.array(df['0'].tolist())
plt.scatter(x, y, alpha=0.8, c='b')
plt.scatter(x2, y2, alpha=0.8, c='r')
#plt.scatter(x2, y2, alpha=0.8, c='y')

#plt.scatter(x, y, alpha=0.8)
#plt.scatter(x, y,  c=colors, alpha=0.8)
plt.show()
