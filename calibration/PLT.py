import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('/media/linwei/disk1/NATS-Bench/350cifar10/r50step.csv', usecols=[1,2,3])
print(df.keys())
#
x = df['0'].tolist()
y1 = df['1'].tolist()
y2 = df['2'].tolist()

fig, ax = plt.subplots(ncols=2, figsize=(10,4))

ax[0].plot(x,y1)
ax[1].plot(x,y2)

fig.show()