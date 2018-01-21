Ad Click Prediction
===
This repository contains the Python scripts needed to predict whether a user will click on an ad.

<a name="sec-data"></a> Data
---
We use the well known [iPinYou dataset][1] to train our classifier. The dataset was created during a Real-Time Bidding 
competition. We use a modified version of this dataset, which contains data for 4,000,000 individual ad bids. For our
classifications, we use the following variables:

| Name | Type |
| --- | --- |
| click | boolean |
| imp | boolean |
| Browser | categorical |
| AdExchange | categorical |
| Adslotvisibility | categorical |
| Adslotformat | categorical |
| Payingprice | int |
| Adslotwidth | int |
| Adslotheight | int |
| interest_* | boolean |
| Inmarket_* | boolean |
| Demographic_gender_female | boolean |

Obviously, `click` is the label variable, which our classifier will regress. `imp` is a logical variable that describes
whether an advertiser is allowed the impression (ie, the advertiser has won with its bid). It is clear that `click` is
directly dependent upon `imp` (ie, `click` cannot be true if `imp` is false). Therefore, we use only bids where
`imp == True` as training and test data.

In our dataset, `Browser` indicates the [user agent][2] of the user targeted by the bid. We implement a simple parser to
map `Browser` to two variables: `OS` indicates the operating system (`Windows`, `Mac OS X`, `Linux` and `other`) and `Browser`
indicates the browser family of the user (`Firefox`, `Chrome`, `Safari`, `Opera`, `Internet Explorer` and `other`).
`AdvertiserID` contains the unique identification number for each advertiser. To reduce our memory overhead, we will use
only bids where `AdvertiserID == 2821` (footware seller).

The actual user-identifiable variables are the `interest_*`-variables. They indicate whether the user has shown a
specific interest into a tag used by iPinYou. For example, if the user is known to be interested in education-related
material, `interest_education == True`. The same holds true for the `Inmarket_*` and
`Demographic_gender_female` variables. We remove the redundant `Demographic_gender_male` variable.

Notice that there are additional variables in the dataset, which we chose to ignore (at least for the initial analysis).
Some of them are completely irrelevant to the `click` prediction, such as the domain name of the ad. Other variables are
discouraged by the distributor of the dataset, because they are not consistent. This holds true for the `conv` variable,
which indicates whether a user actually converted to a customer after clicking on an ad.

The following table lists the discarded variables:

| Name | Type | Reason |
| --- | --- | --- |
| Index | int | Obvious |
| BidID | int | Identifier useless for classification (different for every observation) |
| Time_Bid | int | Bids were conducted only on one afternoon - results wouldn't generalize |
| UserID | int | See `BidID` |
| IP | string | Useless on its own, ISP data has no connection to ads, location data can be extracted, but is already present in `Region` |
| Domain | string | Variable is hashed, referral page of ad probably doesn't exist anymore and has only post-click effects |
| URL | string | See `Domain` |
| AdslotID | int | See `BidID` |
| Adslotfloorprice | int | Information is already contained in `Payingprice` |
| CreativeID | int | Variable is hashed and very specific to the ad |
| Biddingprice | int | Merged into `Payingprice` |
| conv | int | Variable affects only post-click period and is discouraged by provider due to the non-standard conversion definitions of advertisers |
| Region | int | Doesn't generalize beyond Chinese market |
| City | int | See `Region` |
| Demographic_gender_male | bool | Negative `Demographic_gender_female` |

Classification
---
To predict whether a user will click or not is a classical binary classification task. [Data](#sec-data) details that we
have a binary label and binary/categorical features. We figure that to regress `click ~ .`, [random forests][4] are a
perfect fit. They are specifically designed for categorical features. Random forests are ensembles of [decision trees][7]
and therefore reduce the variance of single decision tree algorithms.

We also analyze the dataset using a [logistic regression][5]. This serves as a robust baseline.

We will use [`scikit-learn`][6] and [`statsmodels`][8] to implement the algorithms.

Usage
---
### Notebook
Place the `ads.csv` in the `data` directory. Start `jupyter notebook` and open `click.ipynb`. Follow all steps.
### Scripts
Install dependencies with:

```pip install -r requirements.txt```

Place the `ads.csv` in the `data` directory. Then run `./prepare.py` to clean the dataset and prepare it for training.
It will output `data/ads_clean.csv`. Use `./plot.py` to plot a correlation matrix. Use `./train.py` to train a random
forest or `./train_lr.py` to fit a logistic regression. The help pages display additional information.

Further considerations
---
As seen in [Data](#sec-data), there is a variable `imp` that describes whether a bid from an advertiser has been
successful. It would be interesting to regress `imp` using the other variables. However, we figure that the provided
variables are not conclusive enough to classify `imp` correctly.

[1]: https://arxiv.org/abs/1407.7073
[2]: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent
[3]: https://github.com/ua-parser/uap-python
[4]: https://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf
[5]: https://en.wikipedia.org/wiki/Logistic_regression
[6]: https://scikit-learn.org
[7]: https://en.wikipedia.org/wiki/Decision_tree_learning
[8]: http://www.statsmodels.org/stable/index.html
