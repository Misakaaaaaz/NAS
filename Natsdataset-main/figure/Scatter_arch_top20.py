import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

#df = pd.read_excel('/media/linwei/disk1/NATS-Bench/cifar10-gt935-sum.xlsx',usecols=[0,2,10,11,12,13,14,15])
df = pd.read_excel('/media/linwei/disk1/NATS-Bench/cifar100-gt71-sum.xlsx',usecols=[0,2,10,11,12,13,14,15])
#df = pd.read_excel('/media/linwei/disk1/NATS-Bench/cifar10-gt935-sum.xlsx',usecols=[0,2,10,11,12,13,14,15])


np.random.seed(98)
#print(df['index'].tolist())
#num
print(df.keys())
#cifar10
#df = df[df['ece'] <= 0.038723961]
#cifar100
df = df[df['ece'] <= 0.129214317]
#imagenet
#df = df[df['ece'] <= 0.038723961]

x=df[0].tolist()[:20]

zer = np.zeros(len(x))
#x = zip(zer,x)
#ece
y=df['ece'].tolist()
#params
achlbls0=['|none~0|', '|skip_connect~0|','|avg_pool_3x3~0|', '|nor_conv_1x1~0|', '|nor_conv_3x3~0|']
achlbls1=['|none~1|', '|skip_connect~1|','|avg_pool_3x3~1|', '|nor_conv_1x1~1|', '|nor_conv_3x3~1|']
achlbls2=['|none~2|', '|skip_connect~2|','|avg_pool_3x3~2|', '|nor_conv_1x1~2|', '|nor_conv_3x3~2|']
#arch
archs = df['index'].tolist()
colors = np.random.rand(len(archs))

y= y[:20]
archs = archs[:20]
fig, ax = plt.subplots(ncols=6, sharey=True, figsize=(15, 4))

cmap = mpl.cm.viridis
norm = mpl.colors.BoundaryNorm(archs, cmap.N)
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    #extend='both',
    #extendfrac='auto',
    ticks=archs,
    spacing='uniform',
    orientation='vertical',
    label='Discrete intervals, some other units',
)

ax[0].scatter(x, y, c=archs)
ax[0].set_xticks(achlbls0)

ax[1].scatter(x=df[1].tolist()[:20], y=y, c=archs)
ax[1].set_xticks(achlbls0)

ax[2].scatter(x=df[2].tolist()[:20], y=y, c=archs)
ax[2].set_xticks(achlbls1)
ax[3].scatter(x=df[3].tolist()[:20], y=y, c=archs)
ax[3].set_xticks(achlbls0)
ax[4].scatter(x=df[4].tolist()[:20], y=y, c=archs)
ax[4].set_xticks(achlbls1)
ax[5].scatter(x=df[5].tolist()[:20], y=y, c=archs)
ax[5].set_xticks(achlbls2)




fig.show()