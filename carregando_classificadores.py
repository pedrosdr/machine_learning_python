import pickle
import numpy as np

rn = pickle.load(open('rede_neural_finalizado.sav', 'rb'))
tree = pickle.load(open('arvore_finalizado.sav', 'rb'))
svm = pickle.load(open('svm_finalizado.sav', 'rb'))

with open('credit.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)

x = np.concatenate((xtrain, xtest))

novo_registro = x[19,:].reshape(1,-1)

# Rejeição de algoritmos
prob_tree = tree.predict_proba(novo_registro)
confianca_tree = prob_tree.max()

prob_svm = svm.predict_proba(novo_registro)
confianca_svm = prob_svm.max()

prob_rn = rn.predict_proba(novo_registro)
confianca_rn = prob_rn.max()


# Combinação de classificadores
confianca_minima = 0.9999999

predictions = []
if confianca_tree >= confianca_minima:
    predictions.append(tree.predict(novo_registro)[0])
if confianca_svm >= confianca_minima:
    predictions.append(svm.predict(novo_registro)[0])
if confianca_rn >= confianca_minima:
    predictions.append(rn.predict(novo_registro)[0])
    
paga = len([x for x in predictions if x == 0])
nao_paga = len([x for x in predictions if x == 1])

if paga > nao_paga:
    print('O cliente paga o empréstimo')
elif paga == nao_paga:
    print('Inconclusivo')
else:
    print('O cliente não pagará o empréstimo')
    
