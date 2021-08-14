# CELearn
Recreating SKLearn Library From Scratch


# Documentation

## linear_models.LinearRegression()
- Classic least squares Linear Regression model
- LinearRegression fits a linear model with coefficients w (w1, w2, ..., wn+1) to minimize the residual sum of squares between observed targets and predicted targets, which are predicted by the coefficients.

```python
from celearn.linear_models import LinearRegression

X_train = [[1, 2], [1, 1], [0, 1], [6, 8], [6, 9], [5, 10]]
y_train = [[1], [1], [1], [0], [0], [0]]
X_test = [[1, 0], [0, 0], [7, 8], [6, 7]]
y_test = [[1], [1], [0], [0]]

clf = LinearRegression()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

from celearn.metrics import rmse
rmse(y_hat, y_test)
>>> 0.006649689595207451
```

## neighbors.KNearestNeighbors()
- Classifier model that uses a k-nearest neighbor voting system to classify

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

## ensemble.RandomForest()
- Random forest classifier that fits a number of decision tree classifiers on sub samples of data and uses averaging to improve accuracy, as well as limiting over fitting.

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

## vectorizer.CountVectorizer()
- Returns a document term matrix 

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
