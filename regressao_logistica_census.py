import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with open('census.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)
    
logi = LogisticRegression(random_state=1)
logi.fit(xtrain, ytrain)
predictions = logi.predict(xtest)

print(accuracy_score(ytest, predictions))
print(confusion_matrix(ytest, predictions))
print(classification_report(ytest, predictions))
