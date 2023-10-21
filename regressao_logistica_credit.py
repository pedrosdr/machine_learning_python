import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

with open('credit.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)

lr = LogisticRegression(random_state=1)
lr.fit(xtrain, ytrain)
previsoes = lr.predict(xtest)

print(accuracy_score(ytest, previsoes))
print(classification_report(ytest, previsoes))

cm = ConfusionMatrix(lr)
cm.fit(xtrain, ytrain)
cm.score(xtest, ytest)