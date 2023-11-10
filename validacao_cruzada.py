from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import numpy as np
import seaborn as sns

with open('credit.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)

x = np.concatenate((xtrain, xtest), axis = 0)
y = np.concatenate((ytrain, ytest), axis = 0)

tree_results = []
forest_results = []
knn_results = []
logit_results = []
svm_results = []

for i in range(30):
    print(i)
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    
    tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
    scores = cross_val_score(tree, x, y, cv=kfold)
    tree_results.append(scores.mean())
    
    
for i in range(30):
    print(i)
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    
    rf = RandomForestClassifier(n_estimators=10, criterion="entropy", min_samples_leaf=1, min_samples_split=5)
    scores = cross_val_score(rf, x, y, cv=kfold)
    forest_results.append(scores.mean())
    
    
for i in range(30):
    print(i)
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    
    logit = LogisticRegression(C=1.0)
    scores = cross_val_score(logit, x, y, cv=kfold)
    logit_results.append(scores.mean())

for i in range(30):
    print(i)
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    
    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, x, y, cv=kfold)
    knn_results.append(scores.mean())
    
results = pd.DataFrame({
        'Arvore': tree_results,
        'Random Forest': forest_results,
        'KNN': knn_results,
        'Logistic Regression': logit_results
    })

results.describe()
print(100 * results.std() / results.mean())


# Teste de Normalidade
from scipy.stats import shapiro
alpha = 0.05

print("Tree --> ",shapiro(tree_results))
print("Forest --> ",shapiro(forest_results))
print("KNN --> ",shapiro(knn_results))
print("Logistic Regression --> ",shapiro(logit_results))

sns.displot(forest_results, kind='kde')


# Testes de Hipótese ANOVA e Tukey
from scipy.stats import f_oneway
p = f_oneway(tree_results, forest_results, knn_results, logit_results).pvalue
print(p)

if p <= alpha:
    print('Hipótese nula rejeitada, Dados são diferentes')
else:
    print('Hipótese alternativa rejeitada. Resultados são iguais')

resultados_algoritmos = {
    'accuracy': np.concatenate((tree_results, forest_results, knn_results, logit_results)),
    'algoritmo': [*(['arvore'] * 30),
    *(['floresta'] * 30),
    *(['knn'] * 30),
    *(['logit'] * 30)]
}

from statsmodels.stats.multicomp import MultiComparison
resultados_df = pd.DataFrame(resultados_algoritmos)
multicomp = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])
print(multicomp.tukeyhsd())
multicomp.tukeyhsd().plot_simultaneous()
