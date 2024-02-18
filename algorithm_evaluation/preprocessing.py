import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Preprocessamento
def preprocess_data():
    base:pd.DataFrame = pd.read_csv('house_prices.csv')
    base = base.drop(columns=['id', 'date', 'yr_renovated', 'zipcode'])
    base = base.dropna()

    # Removendo outliers
    plt.hist(base['price'].to_numpy(), bins=10)
    plt.show()
    plt.close()

    plt.boxplot(base['price'])
    plt.show()
    plt.close()

    p3 = base['price'].quantile(0.75)
    p1 = base['price'].quantile(0.25)
    iqr = p3 - p1
    upper_fence = iqr + 1.5 * iqr

    base = base.loc[base['price'] < upper_fence]

    plt.hist(base['price'].to_numpy(), bins=10)
    plt.show()
    plt.close()

    plt.boxplot(base['price'])
    plt.show()
    plt.close()

    # Padronizando as variáveis
    basen = base.to_numpy()
    sc = StandardScaler()
    basen = sc.fit_transform(basen)
    x = basen[:,1:]
    y = basen[:,0:1]

    # Salvando a base tratada
    dfx = pd.DataFrame(x, columns=base.iloc[:,1:].columns)
    dfy = pd.DataFrame(y, columns=['price'])

    dfx.to_csv('house_prices_x.csv', index=False)
    dfy.to_csv('house_prices_y.csv', index=False)



