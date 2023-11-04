import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

with open('census.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)
    
nn = MLPClassifier(
    hidden_layer_sizes=(53,53),
    learning_rate_init=0.001,
    activation='relu',
    solver='adam',
    max_iter=200,
    verbose=True
)

nn.fit(xtrain, ytrain)
predictions = nn.predict(xtest)

print(confusion_matrix(ytest, predictions))
print(accuracy_score(ytest, predictions))
print(classification_report(ytest, predictions))
