import pickle

with open('credit.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)
    
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

rn = MLPClassifier(
    activation='relu',
    batch_size=56,
    solver='adam',
    verbose=True,
    max_iter=1500
)
rn.fit(xtrain, ytrain)

tree = DecisionTreeClassifier(
    criterion='entropy',
    min_samples_leaf=1,
    min_samples_split=5,
    splitter='best'
)
tree.fit(xtrain, ytrain)
# plot_tree(tree)

svm = SVC(
    C=2.0,
    kernel='rbf'
)
svm.fit(xtrain, ytrain)

pickle.dump(rn, open('rede_neural_finalizado.sav', 'wb'))
pickle.dump(tree, open('arvore_finalizado.sav', 'wb'))
pickle.dump(svm, open('svm_finalizado.sav', 'wb'))
