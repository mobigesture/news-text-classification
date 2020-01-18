# News Data Classification
We have a dataset which categorized news into POLITICS, SPORTS, ENTERTAINMENT etc. We have taken this dataset from [here](http://mlg.ucd.ie/datasets/bbc.html). We train our model using [sklearn multiclass](https://scikit-learn.org/stable/modules/multiclass.html) [OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) with 

1. [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
2. [Multinomian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
3. [Linear Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) 

and feature extraction using [TFID Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

# Understanding Data
The data we are using is structured as:

| category | headline | authors | link | short_description | date |
|----------|----------|---------|------|-------------------|------|
| CRIME    | There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV | Melissa Jeltsen | https://www.huffingtonpost.com/entry/texas-amanda-painter-mass-shooting_us_5b081ab4e4b0802d69caad89 | She left her husband. He killed their children. Just another day in America. | 2018-05-26 |

We use `headline` and `short_description` text to train our model with the given `category`.

# Cleaningup Data
We will clean text to remove punctuation marks, stopwords and convert the text to lower case.

# Training and Testing
We will split the data into training set and testing set. We train the model using training set and do prediction on testing set to see the accuracy.

# Validation
We will test our model with the data outside of the dataset and see whether the model is predicting correctly or not.
