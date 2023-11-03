from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open('census.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)
    

svm = SVC(C = 1.0, kernel='linear', random_state=1)
svm.fit(xtrain, ytrain)
predictions = svm.predict(xtest)

print(accuracy_score(ytest, predictions))
print(classification_report(ytest, predictions))
print(confusion_matrix(ytest, predictions))
