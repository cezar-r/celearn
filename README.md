# CELearn
Recreating SKLearn Library From Scratch

# Table of Contents
- linear_models
    - [LinearRegression()](#linear-regression)
    - [LogisticRegression()](#logistic-regression)
    - [SGDClassifier()](#sgd-classifier)
    - [Lasso()](#lasso)
- Neighbors
    - [KNearestNeighbors()](#knn)
- SVM
    - [LinearSVC()](#linear-svc)
- Ensemble
    - [RandomForest()](#random-forests)
- Naive Bayes
    - [GaussianNB()](#gaussian-nb)
- Vectorizer
    - [CountVectorizer()](#count-vectorizer)
    - [TFIDF()](#tfidf)
- Model Selection
    - [train_test_split()](#train-test-split)
- Metrics
    - [mean_squared_error()](#mse)
    - [accuracy_score()](#accuracy)
    - [precision_score()](#precision)
    - [recall_score()](#recall)
    - [f1_score()](#f1)
    - [classification_report()](#classification-report)


# Documentation

## LinearRegression()
`linear_models.LinearRegression(learning_rate = 0.0001, num_epochs = 100000)`
- Classic least squares Linear Regression model
- LinearRegression fits a linear model with coefficients w (w1, w2, ..., wn+1) to minimize the residual sum of squares between observed targets and predicted targets, which are predicted by the coefficients.
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

```python
from celearn.linear_models import LinearRegression

X_train = [[1, 2], [1, 1], [0, 1], [6, 8], [6, 9], [5, 10]]
y_train = [[1], [1], [1], [0], [0], [0]]
X_test = [[1, 0], [0, 0], [7, 8], [6, 7]]
y_test = [[1], [1], [0], [0]]

clf = LinearRegression()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

from celearn.metrics import mean_squared_error
mean_squared_error(y_hat, y_test)
>>> 0.006649689595207451
```
---
## LogisticRegression()
`linear_models.LogisticRegression(learning_rate = 0.0001, num_epochs = 10000)`
- Uses the logisitc regression algorithim to predict classification
- [SKLearn version](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

```python
from celearn.linear_models import LogisticRegression

X_train = [[1, 2], [1, 1], [0, 1], [6, 8], [6, 9], [5, 10]]
y_train = [[1], [1], [1], [0], [0], [0]]
X_test = [[1, 0], [0, 0], [7, 8], [6, 7]]
y_test = [[1], [1], [0], [0]]

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

from celearn.metrics import accuracy_score
accuracy_score(y_hat, y_test)
>>> 1.0
```

## SGDClassifier()
`linear_models.SGDClassifier(learning_rate = 0.0001, num_epochs = 10000)`
- Uses logistic regression to predict classification, with stochastic gradient descent as the optimizer
- [SKLearn version](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) 

```python
from celearn.linear_models import SGDClassifier

X_train = [[1, 2], [1, 1], [0, 1], [6, 8], [6, 9], [5, 10]]
y_train = [[1], [1], [1], [0], [0], [0]]
X_test = [[1, 0], [0, 0], [7, 8], [6, 7]]
y_test = [[1], [1], [0], [0]]

clf = SGDClassifier()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

from celearn.metrics import accuracy_score
accuracy_score(y_hat, y_test)
>>> 1.0
```

## Lasso()
`linear_models.Lasso(learning_rate = 0.0001, num_epochs = 1000, l1_penalty = 800)`
- Linear model that is trained with L1 prior as the regularizer
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

```python
from celearn.linear_models import Lasso

X_train = [[1, 2], [1, 1], [0, 1], [6, 8], [6, 9], [5, 10]]
y_train = [[1], [1], [1], [0], [0], [0]]
X_test = [[1, 0], [0, 0], [7, 8], [6, 7]]
y_test = [[1], [1], [0], [0]]

clf = Lasso()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

from celearn.metrics import mean_squared_error
mean_squared_error(y_hat, y_test)
>>> 0.07778466006128645
```

## KNearestNeighbors()
`neighbors.KNearestNeighbors(neighbors = 3)`
- Classifier model that uses a k-nearest neighbor voting system to classify
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

``` python
from celearn.neighbors import KNearestNeighbors

X_train = [[1, 2], [1, 1], [0, 1], [6, 8], [6, 9], [5, 10]]
y_train = [[1], [1], [1], [0], [0], [0]]
X_test = [[1, 0], [0, 0], [7, 8], [6, 7]]
y_test = [[1], [1], [0], [0]]

clf = KNearestNeighbors()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

from celearn.metrics import accuracy_score
accuracy_score(y_hat, y_test)
>>> 1.0
```

## LinearSVC()
`svm.LinearSVC(pos_label = 1)`
- Classifier model that uses support vector machine to predict classifications. Currently only supports binary classification
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

```python 
from celearn.svm import LinearSVC

X_train = [[1, 2], [1, 1], [0, 1], [6, 8], [6, 9], [5, 10]]
y_train = [[1], [1], [1], [0], [0], [0]]
X_test = [[1, 0], [0, 0], [7, 8], [6, 7]]
y_test = [[1], [1], [0], [0]]

clf = LinearSVC()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

from celearn.metrics import accuracy_score
accuracy_score(y_hat, y_test)
>>> 1.0
```

## RandomForest()
`ensemble.RandomForest(num_trees = 100, num_features = 'auto')`
- Random forest classifier that fits a number of decision tree classifiers on sub samples of data and uses averaging to improve accuracy, as well as limiting over fitting.
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

```python
from celearn.ensemble import RandomForest

X_train = [[1, 2], [1, 1], [0, 1], [6, 8], [6, 9], [5, 10]]
y_train = [[1], [1], [1], [0], [0], [0]]
X_test = [[1, 0], [0, 0], [7, 8], [6, 7]]
y_test = [[1], [1], [0], [0]]

clf = RandomForest()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

from celearn.metrics import accuracy_score
accuracy_score(y_hat, y_test)
>>> 1.0
```

## GaussianNB()
`naive_bayes.GaussianNB()`
- Uses the naive bayes algorithim to predict classifications with the gaussian probability density function.
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

```python
from celearn.naive_bayes import GaussianNB

X_train = [[1, 2], [1, 1], [0, 1], [6, 8], [6, 9], [5, 10]]
y_train = [[1], [1], [1], [0], [0], [0]]
X_test = [[1, 0], [0, 0], [7, 8], [6, 7]]
y_test = [[1], [1], [0], [0]]

clf = GaussianNB()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

from celearn.metrics import accuracy_score
accuracy_score(y_hat, y_test)
>>> 1.0
```

## CountVectorizer()
`vectorizer.CountVectorizer()`
- Returns a document term matrix 
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

```python
from celearn.vectorizer import CountVectorizer

corpus =[['This is a dog and this is a dog'],
         ['This is a cat']]
         
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

vectorizer.get_feature_names()
>>> ['and', 'a', 'cat', 'dog', 'this', 'is']
X
>>> [[1, 2, 0, 2, 2, 2], [0, 1, 1, 0, 1, 1]]
```

## vectorizer.TFIDF()
`vectorizer.TFIDF()`
- Returns a TFIDF as dict, pandas DataFrame, or numpy matrix
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

```python
from celearn.vectorizer import TFIDF

corpus =[['This is a dog and this is a dog'],
         ['This is a cat'],
         ['This is a frog']]

labels = [[0], 
          [1],
          [1]]
          
tfidf = TFIDF(retval = 'df')
df = tfidf.fit(corpus, labels)
df
>>>     is  this  and  frog  dog    a  cat
>>>  0  0.5   0.5  1.0   0.0  1.0  0.5  0.0
>>>  1  0.5   0.5  0.0   1.0  0.0  0.5  1.0
df.to_numpy()
>>> [[0.5 0.5 1.  0.  1.  0.5 0. ]
>>>  [0.5 0.5 0.  1.  0.  0.5 1. ]]
```

## train_test_split()
`model_selection.train_test_split(X, y, test_size = .25)
- Splits X and y into training and testing data. Default test size is .25
- [SKLearn version](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

```python
from celearn.model_selection import train_test_split

X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y = [[1], [0], [1]]

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train, y_train
>>> ([4, 5, 6], [7, 8, 9])
>>> ([0], [1])
X_test, y_test
>>> ([1, 2, 3],)
>>> ([1],)
```

## mean_squared_error()
`metrics.mean_squared_error(y_hat, y_test)`
- Returns the mean squared error between two lists
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)

```python
from celearn.metrics import mean_squared_error

y_test = [[1], [2], [3], [4], [1], [2], [3]]
y_pred = [[1], [2], [3], [3], [1], [2], [3]]

mean_squared_error(y_hat, y_test)
>>> 0.14
```

## accuracy_score()
`metrics.accuracy_score(y_hat, y_test)`
- Returns the accuracy of the predicted labels
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

```python
from celearn.metrics import accuracy_score

y_test = [[1], [2], [3], [4], [1], [2], [3]]
y_pred = [[1], [2], [3], [3], [1], [2], [3]]

accuracy_score(y_test, y_pred)
>>> 0.86
```

## precision_score()
`metrics.precision_score(y_hat, y_test)`
- Returns the precision of the predicted labels. Currently only supports binary classification
- Uses (TP/(TP + FP)) to calculate precision.
- [SKLearn Version](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

```python
from celearn.metrics import precision_score

y_test = [[1], [0], [1], [1], [0]]
y_pred = [[1], [1], [1], [1], [0]]

precision_score(y_hat, y_test)
>>> 0.8
```

## recall_score()
`metrics.recall_score(y_hat, y_test)`
- Returns the recall of the predicted labels. Currently only supports binary classification
- Uses (TP/(TP + FN)) to calculate recall
- [SKLearn version](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

```python
from celearn.metrics import recall_score

y_test = [[1], [0], [1], [1], [0]]
y_pred = [[1], [1], [1], [1], [0]]

recall_score(y_hat, y_test)
>>> 1.0
```

## f1_score()
`metrics.f1_score(y_hat, y_test)`
- Returns the weighted average score between recall and precision. Currently only supports binary classification
- Uses (2 * (precision * recall) / (precision + recall)) to calculate F1
- [SKLearn version](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

```python
from celearn.metrics import f1_score

y_test = [[1], [0], [1], [1], [0]]
y_pred = [[1], [1], [1], [1], [0]]

f1_score(y_hat, y_test)
>>> .89
```

## classification_report()
`metrics.classification_report(y_hat, y_test)`
- Returns a report on the predicted labels. Report includes precision, recall, f1-score and support
- Report can be accessed in dictionary format by accessing the `.as_dict` attribute
- [SKLearn version](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

```python
from celearn.metrics import classification_report

y_test = [[1], [2], [3], [4], [1], [2], [3]]
y_pred = [[1], [2], [3], [3], [1], [2], [3]]

report = classification_report(y_hat, y_test)
report 
>>>      labels    precision       recall     f1-score      support
>>>
>>>           1          1.0          1.0          1.0            2
>>>           2          1.0          1.0          1.0            2
>>>           3         0.86          1.0         0.92            2
>>>           4          1.0         0.86         0.92            1
report.as_dict
>>> {1: {'precision': 1.0, 
>>>      'recall': 1.0, 
>>>      'f1-score': 1.0, 
>>>      'support': 2}, 
>>>  2: {'precision': 1.0, 
>>>      'recall': 1.0, 
>>>      'f1-score': 1.0, 
>>>      'support': 2}, 
>>>  3: {'precision': 0.86, 
>>>      'recall': 1.0, 
>>>      'f1-score': 0.92, 
>>>      'support': 2}, 
>>>  4: {'precision': 1.0, 
>>>      'recall': 0.86, 
>>>      'f1-score': 0.92, 
>>>      'support': 1}}
```
