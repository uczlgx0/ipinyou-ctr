from pandas import get_dummies, read_pickle
import statsmodels.api as sm

ads = read_pickle('data/ads_clean.p')
browser_dummies = get_dummies(ads['Browser'], prefix='Browser')
browser_dummies.drop(['Browser_other'], axis=1, inplace=True)
adex_dummies = get_dummies(ads['AdExchange'], prefix='AdExchange')
adex_dummies.drop(['AdExchange_1'], axis=1, inplace=True)
advi_dummies = get_dummies(ads['Adslotvisibility'], prefix='Adslotvisibility')
advi_dummies.drop(['Adslotvisibility_FifthView'], axis=1, inplace=True)
adfo_dummies = get_dummies(ads['Adslotformat'], prefix='Adslotformat')
adfo_dummies.drop(['Adslotformat_Na'], axis=1, inplace=True)
os_dummies = get_dummies(ads['OS'], prefix='OS')
os_dummies.drop(['OS_other'], axis=1, inplace=True)
ads = ads.join([browser_dummies, adex_dummies, advi_dummies, adfo_dummies, os_dummies])
ads.drop(['Browser', 'AdExchange', 'Adslotvisibility', 'Adslotformat', 'OS'], axis=1, inplace=True)
ads['intercept'] = 1.0

ads = ads.astype(int)
logit = sm.Logit(ads['click'], ads.drop(['click'], axis=1))
result = logit.fit(maxiter=100)
