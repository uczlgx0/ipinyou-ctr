#!/usr/bin/env python3
import pandas as pd
from ua_parser.user_agent_parser import ParseUserAgent

use_browser = False
ads = pd.read_csv('data/ads.csv', usecols=['click', 'AdvertiserID', 'AdExchange', 'Adslotwidth',
                                           'Adslotheight', 'imp', 'interest_news',
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
                                           'Demographic_gender_male', 'Demographic_gender_famale'] + (['Browser'] if use_browser else []))
cols = ['click'] + [col for col in ads if col != 'click']
ads = ads[cols]
#ads.drop(['Unnamed: 0'], axis=1, inplace=True)
ads.rename(
    columns={'interest_eduation': 'interest_education', 'Demographic_gender_famale': 'Demographic_gender_female'},
    inplace=True)
ads.dropna(inplace=True)  # ToDo: Decrease strictness to preserve more positive classes
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
ads[boolean_cols] = ads[boolean_cols].astype(bool)
int_cols = ['AdExchange', 'AdvertiserID', 'Adslotwidth', 'Adslotheight']
ads[int_cols] = ads[int_cols].astype(int)
if use_browser:
    ads['Browser'] = ads['Browser'].map(lambda x: ParseUserAgent(x)['family'])
    ads['Browser'] = ads['Browser'].astype('category')
ads.to_csv('data/ads_clean.csv')
ads.head(10)
ads.info()
ads.describe()
print(len(ads[ads['imp'] == 0]))
