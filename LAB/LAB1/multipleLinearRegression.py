import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd



data = pd.read_csv("iris.csv")



x1 = data.lol

x2 = data.temp

y_tmp = data.cnt



x = []

y = []



for i in range(len(x1)):

    x.append([x1[i], x2[i]])

    y.append([y_tmp[i]])



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)



LinReg = linear_model.LinearRegression()



LinReg.fit(x_train, y_train)

y_pred = LinReg.predict(x_test)



print("The Linear Regression Coefficients:", LinReg.coef_)

print("The Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))

print("The R Squared Score:", round(r2_score(y_test, y_pred), 2))





fig = plt.figure()

ax = fig.add_subplot(111,)

ax.scatter(x1, y, c="blue", marker="o", alpha=0.5)

ax.set_xlabel('sepal length')

#ax.set_ylabel('sepal width')

ax.set_ylabel('petal length')

plt.show()