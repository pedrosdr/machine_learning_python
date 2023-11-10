import pickle
import numpy as np

rn = pickle.load(open('rede_neural_finalizado.sav', 'rb'))
tree = pickle.load(open('arvore_finalizado.sav', 'rb'))
svm = pickle.load(open('svm_finalizado.sav', 'rb'))

with open('credit.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)

x = np.concatenate((xtrain, xtest))

novo_registro = x[19,:].reshape(1,-1)

predictions = []
predictions.append(rn.predict(novo_registro)[0])
predictions.append(tree.predict(novo_registro)[0])
predictions.append(svm.predict(novo_registro)[0])

# Combinação de classificadores
paga = len([x for x in predictions if x == 0])
nao_paga = len([x for x in predictions if x == 1])

if paga > nao_paga:
    print('O cliente paga o empréstimo')
else:
    print('O cliente não pagará o empréstimo')
