from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np

with open('credit.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)
    

x = np.concatenate((xtrain, xtest), axis = 0)
y = np.concatenate((ytrain, ytest), axis = 0)


# # Decision Tree
# params = {
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 5, 10]
# }

# gs = GridSearchCV(DecisionTreeClassifier(), params)
# gs.fit(x, y)
# best_params = gs.best_params_
# best_score = gs.best_score_


# # Random Forest
# params = {
#     'criterion': ['gini', 'entropy', 'log_loss'],
#     'n_estimators': [10, 40, 100, 150],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 5, 10]
# }

# gs = GridSearchCV(RandomForestClassifier(), params)
# gs.fit(x, y)
# best_params = gs.best_params_
# best_score = gs.best_score_

# # KNN
# params = {
#     'n_neighbors': [3, 5, 10, 20],
#     'p': [1, 2, 3]
# }

# gs = GridSearchCV(KNeighborsClassifier(), params)
# gs.fit(x, y)
# best_params = gs.best_params_
# best_score = gs.best_score_

# Logistic Regression
# params = {
#     'tol': [0.001, 0.0001, 0.00001, 0.000001],
#     'C': [1.0, 1.5, 2.0],
#     'solver': ['lbfgs', 'sag', 'saga']
# }

# gs = GridSearchCV(LogisticRegression(), params)
# gs.fit(x, y)
# best_params = gs.best_params_
# best_score = gs.best_score_

# SVM
params = {
    'tol': [0.001, 0.0001, 0.00001, 0.000001],
    'C': [1.0, 1.5, 2.0],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

gs = GridSearchCV(SVC(), params)
gs.fit(x, y)
best_params = gs.best_params_
best_score = gs.best_score_
