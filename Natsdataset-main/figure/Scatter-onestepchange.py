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
df = pd.read_csv('../Figure_data/cifar10_ecesum.csv', usecols=[2,3,4])
#df = pd.read_csv('/media/linwei/disk1/NATS-Bench/NATS-details/ImageNet16-120-4binspre&post-sum.csv', usecols=cols)

print(df.keys())
onestep=[6111,2581,4954,13714,2778,582,9586,5292,2030,655,81,3456,7570,13403,8452,12687,13981,1462]
df1 = df[df['5bins-ece-pre'].isin(onestep)]

#ece
x=df['5bins-adaece-pre'].tolist()

x2=df1['5bins-adaece-pre'].tolist()
#testacc
y=df['5bins-ece-post'].tolist()

y2=df1['5bins-ece-post'].tolist()

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
