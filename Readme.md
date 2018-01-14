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
| Browser | string |
| AdExchange | int |
| AdvertiserID | int |
| Adslotwidth | int |
| Adslotheight | int |
| interest_* | boolean |
| Inmarket_* | boolean |
| Demographic_gender_male | boolean |
| Demographic_gender_female | boolean |

Obviously, `click` is the label variable, which our classifier will regress. `imp` is a logical variable that describes
whether an advertiser is allowed the impression (ie, the advertiser has won with its bid). It is clear that `click` is
directly dependent upon `imp` (ie, `click` cannot be true if `imp` is false). Therefore, we will use only bids where
`imp == True` as training and test data.

In our dataset, `Browser` indicates the [user agent][2] of the user targeted by the bid. We use the Python package
[ua-parser][3] to translate the user agent into a factor variable which will indicate the browser family of the user
(ie, `Mozilla Firefox`, `Chrome`, `Safari`, ...). `AdvertiserID` contains the unique identification number for each
advertiser. To reduce our memory overhead, we will use only bids where `AdvertiserID == 2259` (a milk provider).

The actual user-identifiable variables are the `interest_*`-variables. They indicate whether the user has shown a
specific interest into a tag used by iPinYou. For example, if the user is known to be interested in education-related
material, `interest_education == True`. The same holds true for the `Inmarket_*`, `Demographic_gender_male` and
`Demographic_gender_female` variables.

Notice that there are additional variables in the dataset, which we chose to ignore (at least for the initial analysis).
Some of them are completely irrelevant to the `click` prediction, such as the domain name of the ad. Others could be
interesting for the `click` prediction and may be used in future predictions. Examples include the ad format or the
region of the user. We decided against using ad format variables like `Adviewability` because they contain too many
missing values. Other variables are discouraged by the distributor of the dataset, because they are not consistent.
This holds true for the `conv` variable, which indicates whether a user actually converted to a customer after clicking
on an ad.

Classification
---
To predict whether a user will click or not is a classical binary classification task. [Data](#sec-data) details that we
have a binary label and binary/categorical features. We figure that to regress `click ~ .`, [random forests][4] are a
perfect fit. They are applicable to categorical features, as opposed to other Machine Learning algorithms that are only
applicable to numerical features (eg SVM). Random forests are ensembles of [decision trees][7] and therefore reduce the
variance of single decision tree algorithms.

We may also analyze the dataset using a [logistic regression][5]. This could serve as a robust baseline.

We will use [`scikit-learn`][6] to implement the algorithms.

Usage
---
Place the `ads.csv` in the `data` directory. Then run `./prepare.py` to clean the dataset and prepare it for training.
It will output `data/ads_clean.csv`. Use `./plot.py` to plot a correlation matrix. `./train.py` is still under
development.

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