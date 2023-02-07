import pandas as pd
import numpy as np

df1 = pd.read_csv('/media/linwei/disk1/NATS-Bench/tss-cifar10-gt935-0t200.csv', usecols=[1, 2])
df2 = pd.read_csv('/media/linwei/disk1/NATS-Bench/tss-cifar10-gt935-200t417.csv', usecols=[1, 2])

#print(df1['0'])
ecelist = np.array(df1['0'])
ecelist = np.concatenate([ecelist, np.array(df2['0'])])

#acclist = np.array(df1['1'])
#acclist += np.array(df2['1'])

print(len(ecelist))

print("ECEMEAN: {:.4f}  ECESTD:{:.4f}".format(np.mean(np.array(ecelist)), np.std(np.array(ecelist))))