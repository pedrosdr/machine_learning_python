from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open('credit.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)
    
# df = pd.DataFrame({
#         'x': xtrain[:,0],
#         'y': xtrain[:,1],
#         'class': ytrain,
#     })


# ['green' if x == 1 else 'red' for x in ytrain]

# class_colors = {0: 'red', 1: 'green'}
# df['colors'] = df['class'].map(class_colors)
    
# plt.scatter(xtrain[:,0], xtrain[:,1], color = df['colors'])

svm = SVC(C = 2.0, kernel='rbf', random_state=1)
svm.fit(xtrain, ytrain)
predictions = svm.predict(xtest)

print(accuracy_score(ytest, predictions))
print(classification_report(ytest, predictions))
print(confusion_matrix(ytest, predictions))
