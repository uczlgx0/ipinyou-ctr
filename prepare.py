#!/usr/bin/env python3
import pandas as pd

use_browser = False
ads = pd.read_csv('data/ads.csv', low_memory=False, usecols=['click', 'Browser' 'AdvertiserID', 'AdExchange', 'Adslotwidth',
                                           'Adslotheight', 'Adslotvisibility', 'Adslotformat', 'Biddingprice' 'imp', 'interest_news',
                                           'interest_eduation', 'interest_automobile', 'interest_realestate',
                                           'interest_IT', 'interest_electronicgame', 'interest_fashion',
                                           'interest_entertainment', 'interest_luxury', 'interest_homeandlifestyle',
                                           'interest_health', 'interest_food', 'interest_divine',
                                           'interest_motherhood_parenting', 'interest_sports',
                                           'interest_travel_outdoors',
                                           'interest_social', 'interest_art_photography_design',
                                           'interest_onlineliterature', 'interest_3c', 'interest_culture',
                                           'interest_sex',
                                           'Inmarket_3cproduct', 'Inmarket_appliances', 'Inmarket_clothing_shoes_bags',
                                           'Inmarket_Beauty_PersonalCare', 'Inmarket_infant_momproducts',
                                           'Inmarket_sportsitem', 'Inmarket_outdoor', 'Inmarket_healthcareproducts',
                                           'Inmarket_luxury', 'Inmarket_realestate', 'Inmarket_automobile',
                                           'Inmarket_finance', 'Inmarket_travel', 'Inmarket_education',
                                           'Inmarket_service', 'Inmarket_electronicgame', 'Inmarket_book',
                                           'Inmarket_medicine', 'Inmarket_food_drink', 'Inmarket_homeimprovement',
                                           'Demographic_gender_male', 'Demographic_gender_famale', 'Payingprice'])
if 'Unnamed: 0' in ads:
    ads.drop('Unnamed: 0', axis=1, inplace=True)
ads.dropna(subset=['click'], inplace=True)
cols = ['click'] + [col for col in ads if col != 'click']
ads = ads[cols]
ads.rename(
    columns={'interest_eduation': 'interest_education', 'Demographic_gender_famale': 'Demographic_gender_female'},
    inplace=True)

boolean_cols = ['imp', 'click', 'interest_news',
                'interest_education', 'interest_automobile', 'interest_realestate',
                'interest_IT', 'interest_electronicgame', 'interest_fashion',
                'interest_entertainment', 'interest_luxury', 'interest_homeandlifestyle',
                'interest_health', 'interest_food', 'interest_divine',
                'interest_motherhood_parenting', 'interest_sports', 'interest_travel_outdoors',
                'interest_social', 'interest_art_photography_design',
                'interest_onlineliterature', 'interest_3c', 'interest_culture', 'interest_sex',
                'Inmarket_3cproduct', 'Inmarket_appliances', 'Inmarket_clothing_shoes_bags',
                'Inmarket_Beauty_PersonalCare', 'Inmarket_infant_momproducts',
                'Inmarket_sportsitem', 'Inmarket_outdoor', 'Inmarket_healthcareproducts',
                'Inmarket_luxury', 'Inmarket_realestate', 'Inmarket_automobile',
                'Inmarket_finance', 'Inmarket_travel', 'Inmarket_education',
                'Inmarket_service', 'Inmarket_electronicgame', 'Inmarket_book',
                'Inmarket_medicine', 'Inmarket_food_drink', 'Inmarket_homeimprovement',
                'Demographic_gender_male', 'Demographic_gender_female']
ads[boolean_cols] = ads[boolean_cols].fillna(0)
ads[boolean_cols] = ads[boolean_cols].astype(bool)
ads = ads[ads['imp']]
ads.drop(['imp'], axis=1, inplace=True)
ads.loc[ads['Payingprice'].isnull(), 'Payingprice'] = ads.loc[ads['Payingprice'].isnull(), 'Biddingprice']
ads.drop('Biddingprice', axis=1, inplace=True)
# ToDo: Use only AdvertiserID == 2821
ads.drop(['AdvertiserID'], axis=1, inplace=True)
ads['AdExchange'] = ads['AdExchange'].astype('int').astype('category')
ads['Adslotvisibility'] = ads['Adslotvisibility'].astype('category')
ads['Adslotformat'] = ads['Adslotformat'].astype('category')
ads['Browser'] = ads['Browser'].astype(str)


def map_browser(agent):
    browsers = ['edge', 'trident', 'chrome', 'firefox', 'safari', 'opera']
    for browser in browsers:
        if browser in agent.lower():
            return 'ie' if browser == 'trident' else browser
    return 'other'


def map_os(agent):
    os_list = ['windows', 'linux', 'mac os x']
    for os in os_list:
        if os in agent.lower():
            return os
    return 'other'


ads['OS'] = ads['Browser'].map(lambda x: map_os(x), na_action=None)
ads['Browser'] = ads['Browser'].map(lambda x: map_browser(x), na_action=None)
# ToDo: Make dummy variables for categoricals
ads.dropna(inplace=True)  # ToDo: Decrease strictness to preserve more positive classes
int_cols = ['AdvertiserID', 'Adslotwidth', 'Adslotheight']
ads[int_cols] = ads[int_cols].astype(int)
ads.to_csv('data/ads_clean.csv', index=False)
ads.to_pickle('data/ads_clean.p')
ads.head(10)
ads.info()
ads.describe()
