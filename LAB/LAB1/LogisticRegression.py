import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split

from sklearn import metrics

import pandas as pd



data = pd.read_csv("iris.csv")



cols = ['lol', 'temp', 'cnt', 'end']



x = data[cols]

y = data.Irissetosa



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)



LogReg = LogisticRegression()

LogReg.fit(x_train, y_train)



y_pred = LogReg.predict(x_test)



confusionMatrix = metrics.confusion_matrix(y_test, y_pred)

print("The confusion matrix:\n", confusionMatrix)

print("The accuracy score:", metrics.accuracy_score(y_test, y_pred))

#print("The precision score:", metrics.precision_score(y_test, y_pred))

#print("The recall score:", metrics.recall_score(y_test, y_pred))



"""

class_names = [0, 1]

fig, ax = plt.subplots()

tick_marks = np.arange(2)

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)



sns.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap="YlGnBu", fmt="g")

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()"""