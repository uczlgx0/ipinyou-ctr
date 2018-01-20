#!/usr/bin/env python3
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import makedirs
from os.path import join, isdir
from pickle import dump
from time import time


from pandas import get_dummies, read_pickle
from scipy import stats


parser = ArgumentParser('Fits Logit on data', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output', type=str, default='dump', help='Output directory')
parser.add_argument('-m', '--mode', type=str, default='stat', choices=['rfe', 'cv', 'stat'], help='Mode to use')
args = parser.parse_args()
dump_dir = args.output
mode = args.mode

ads = read_pickle('data/ads_clean.p')
ads.drop(['AdvertiserID'], axis=1, inplace=True)
browser_dummies = get_dummies(ads['Browser'], prefix='Browser')
browser_dummies.drop(['Browser_other'], axis=1, inplace=True)
adex_dummies = get_dummies(ads['AdExchange'], prefix='AdExchange')
adex_dummies.drop(['AdExchange_1'], axis=1, inplace=True)
advi_dummies = get_dummies(ads['Adslotvisibility'], prefix='Adslotvisibility')
advi_dummies.drop(['Adslotvisibility_Na'], axis=1, inplace=True)
adfo_dummies = get_dummies(ads['Adslotformat'], prefix='Adslotformat')
adfo_dummies.drop(['Adslotformat_Na'], axis=1, inplace=True)
os_dummies = get_dummies(ads['OS'], prefix='OS')
os_dummies.drop(['OS_other'], axis=1, inplace=True)
ads = ads.join([browser_dummies, adex_dummies, advi_dummies, adfo_dummies, os_dummies])
ads.drop(['Browser', 'AdExchange', 'Adslotvisibility', 'Adslotformat', 'OS'], axis=1, inplace=True)


if mode == 'rfe':
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    logit_rfe = LogisticRegression()
    rfe = RFECV(logit_rfe, cv=5)
    rfe = rfe.fit(ads.drop(['click'], axis=1), ads['click'])
    print(rfe.support_)
    result = rfe
elif mode == 'stat':
    import statsmodels.api as sm
    ads['intercept'] = 1.0
    ads = ads.astype(int)
    logit = sm.Logit(ads['click'], ads.drop(['click'], axis=1))
    result = logit.fit()
else:
    print('Implement CV')

path = join(dump_dir, 'lr')
if not isdir(path):
    makedirs(path)
path = join(path, 'lr_' + str(time()) + '.p')
with open(path, 'wb') as file:
    dump(result, file)

# Workaround for https://github.com/statsmodels/statsmodels/issues/3931
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
print(result.summary())
