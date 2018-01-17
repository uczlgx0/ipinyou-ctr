#!/usr/bin/env python3
import pandas as pd
from matplotlib import pyplot as plt

ads = pd.read_pickle('data/ads_clean.p')
corr = ads.corr()
plt.matshow(corr)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.savefig('img/corr.png')
