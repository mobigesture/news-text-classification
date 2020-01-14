import re
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# Read json file
df = pd.read_json("./News_Category_Dataset_v2.json",lines=True)

#create text field with headline and short_description
df['text'] = df['headline'] + ' '+ df['short_description']

#clean text
df['text'] = df['text'].map(lambda com : clean_text(com))

#split data into train and test
train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)

# get train and test, text values
X_train = train['text'].values
X_test  = test['text'].values

# get train and test, category labels
Y_train=train['category'].values
Y_test=test['category'].values

# create a pipeline with TfidVectorizer, OneVsRestClassifier and logistic regression
LREG_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words,use_idf=True, max_df=0.95)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'))),
            ])
# fit and predict
LREG_pipeline.fit(X_train, Y_train)
pred = LREG_pipeline.predict(X_test)
print('Accuracy score : %.3f ' % accuracy_score(Y_test, pred))