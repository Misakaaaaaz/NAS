from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")
cols = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
df = pd.read_excel('/media/linwei/disk1/NATS-Bench/imagenet-bins.xlsx', usecols=cols)
#df = pd.read_csv('/media/linwei/disk1/NATS-Bench/NATS-details/cifar100-4binspre&post-sum.csv', usecols=cols)
#df = pd.read_csv('/media/linwei/disk1/NATS-Bench/NATS-details/ImageNet16-120-4binspre&post-sum.csv', usecols=cols)

print(df)
# Generate a large random dataset
rs = np.random.RandomState(33)
#d = pd.DataFrame(data=rs.normal(size=(100, 26)),
#                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = df.corr()
print(corr)

# Generate a mask for the upper triangle
#mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
#cmap = sns.dark_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,  cmap=cmap, vmax=1., vmin=-1.,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


f.show()
#@@结论是tempearature后模型的corr不同数据集上存在明显差异，越复杂的数据集上temp前后的分布相关性越低