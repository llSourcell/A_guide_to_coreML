# Required imports
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import coremltools

# Reading in and parsing data
raw_data = open('SMSSpamCollection.txt', 'r')
sms_data = []
for line in raw_data:
    split_line = line.split("\t")
    sms_data.append(split_line)

# Splitting data into messages and labels and training and test
sms_data = np.array(sms_data)
X = sms_data[:, 1]
y = sms_data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=22)

# Building Pipelines
pipeline_1 = Pipeline([('vect', CountVectorizer()),('clf', MultinomialNB())])
pipeline_2 = Pipeline([('vect', TfidfVectorizer()),('clf', MultinomialNB())])
pipeline_3 = Pipeline([('vect', CountVectorizer()),('clf', LinearSVC())])
pipeline_4 = Pipeline([('vect', TfidfVectorizer()),('clf', LinearSVC())])
pipeline_5 = Pipeline([('vect', CountVectorizer()),('clf', RandomForestClassifier())])
pipeline_6 = Pipeline([('vect', TfidfVectorizer()),('clf', RandomForestClassifier())])
pipelines = [pipeline_1, pipeline_2, pipeline_3, pipeline_4, pipeline_5, pipeline_6]

# Performing classification and calculating accuracy
for pipeline in pipelines:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# Creating and saving an .mlmodel file and a list of words
vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(X)
words = open('words_ordered.txt', 'w')
for feature in vectorizer.get_feature_names():
    words.write(feature.encode('utf-8') + '\n')
words.close()
model = LinearSVC()
model.fit(vectorized, y)
coreml_model = coremltools.converters.sklearn.convert(model, "message", 'label')
coreml_model.save('MessageClassifier.mlmodel')
