# Natural Language Processing Project: Twitter Sentiment Classification

Author: [Morgan Nash](mailto:morganmichellenash@gmail.com)

November 2025

<img src="images/global_digital_network.jpg" width="800">

# Introduction

This project uses Natural Language Processing (NLP) and Classification Methods to predict the sentiment of a Tweet based on its content.

# Business Understanding

The presence of social media today allows for constant real-time communication which can present issues for companies to deal with. [Twitter](https://x.com/), now X, is one of the largest social media platforms where users' short messages are posted in real-time. One negative tweet could quickly blow up into a company's PR nightmare if left untouched. Ineffective monitoring of what's being said about a company online can lead to slow crisis response, missed customer service opportunities, and permanent brand damage. A classification model is necessary to filter out the noise and help to automate some of this process.

<img src="images/domino_effect.jpg" width="600">

### Stakeholders:

The final model acts as a first-pass filtering system that can help several important company teams:

**Customer Service Teams** can jump on a addressing negative tweets, rather than leaving them to get more attention.

**Marketing Teams** can use the findings to figure out what people are saying about the company brand. They can also use the positive tweets for promotion.


# Data Understanding

The dataset used for this project comes from [CrowdFlower Open Source Datasets](https://data.world/crowdflower/brands-and-product-emotions) and contains over 9000 tweets from 2013 that reference Apple or Google products. The tweets were related to the [South by Southwest festival (SXSW)](https://www.sxsw.com/) and were rated by humans as to whether the tweets expressed positive, negative, no emotion towards a brand or product, or they couldn't tell. I combined the 'no emotion..' and 'can't tell' into a 'neutral' sentiment. The dataset originally contained 3 columns which I renamed for clarity: tweet, product_name, and sentiment.

## Limitations



### Sentiment Distribution

Neutral  60.98%

Positive 32.75%

Negative  6.27%

<img src="images/sentiment_distribution.jpg" width="800">


## Data Preparation:

To prepare the data, I performed basic cleaning including removal of duplicates, removal of null values, renamed columns for clarity and standardized the text case. I used Regex patterns to remove hashtags, urls, htmls, punctuation, etc. I created two custom stopwords lists. The first included certain words from NLTK's 'english' list and a couple of dataset specific words. The second was more extensive, combining the entire 'english' stopwords list from NLTK with more sentiment-neutral topic noise words like 'apple' and 'google'. Excluding these words from the features helped the model to ignore some of the noise. I also used word_tokenize from NLTK to tokenize the text data for exploratory analysis.

# Exploratory Data Analysis

<img src="images/initial_sentiment_distribution.jpg" width="800">

<img src="images/updated_sentiment_distribution.jpg" width="800">





<img src="images/top_words_before.jpg" width="1000">

<img src="images/top_words_after.jpg" width="1000">

<img src="images/top_words_after_2.jpg" width="1000">


<img src="images/top_negative_predictors.jpg" width="1000">

<img src="images/top_positive_predictors.jpg" width="1000">



# Conclusion



## Recommendations

## Next Steps
