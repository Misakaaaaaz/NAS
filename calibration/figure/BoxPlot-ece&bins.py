import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="ticks", palette="pastel")
url='/media/linwei/disk1/NATS-Bench/NATS-details/cifar100-4binspre&post-sum.csv'
#cols = [2,8,14,20]
df = pd.read_csv(url, usecols=[4,10,16,22])
df2 = pd.read_csv(url, usecols=[4,5,6,7])

f, ax = plt.subplots(1,2,figsize=(11, 9))
sns.boxplot(data=df, ax=ax[0])
sns.boxplot(data=df2, ax=ax[1])
f.show()