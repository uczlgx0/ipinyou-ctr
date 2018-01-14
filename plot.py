#!/usr/bin/env python3
import pandas as pd
from matplotlib import pyplot as plt

ads = pd.read_csv('data/ads_clean.csv')
corr = ads.corr('kendall')
plt.matshow(corr['click', 'imp'])
plt.yticks(range(len(corr.columns)), corr.columns)
plt.savefig('img/corr.png')
