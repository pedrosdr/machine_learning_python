import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

with open('credit.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)
    
    
nn = MLPClassifier(
    max_iter=1500, 
    verbose=True,
    tol=0.00001,
    solver='adam',
    activation='relu',
    hidden_layer_sizes=(2, 2),
    learning_rate_init=0.001
)

nn.fit(xtrain, ytrain)
predictions = nn.predict(xtest)

print(confusion_matrix(ytest, predictions))
print(accuracy_score(ytest, predictions))
print(classification_report(ytest, predictions))
