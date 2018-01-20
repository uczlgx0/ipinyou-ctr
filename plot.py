#!/usr/bin/env python3
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

ads = pd.read_pickle('data/ads_clean.p')
corr = ads.corr()
fig, ax = plt.subplots(figsize=(30,30))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax)
plt.savefig('img/corr.png')
plt.show()