import pandas as pd
from apyori import apriori

base = pd.read_csv('mercado1.csv', header=None)

transactions = []   
for i in range(len(base)):
    transactions.append([str(base.iloc[i, j]) for j in range(4)])
    

rules = list(apriori(transactions, min_support = 0.3, min_confidence = 0.8, min_lift = 2))

rules_dict = {
    'a': [],
    'b': [],
    'support': [],
    'confidence': [],
    'lift': []
}

for rule in rules:
    for r in rule[2]:
        rules_dict['a'].append(list(r[0]))
        rules_dict['b'].append(list(r[1]))
        rules_dict['support'].append(rule[1])
        rules_dict['confidence'].append(r[2])
        rules_dict['lift'].append(r[3])

df_rules = pd.DataFrame(rules_dict)
    