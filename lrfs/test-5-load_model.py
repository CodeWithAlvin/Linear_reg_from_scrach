import linearRegression
from sklearn import datasets
import numpy as np

model = linearRegression.LinearRegression()
data = datasets.load_boston()
x = data.data
y = data.target

model.load("boston.mdl")

prediction = model.predict(x).flatten()

print(prediction,y)

print('loss = ',sum(abs(y-prediction))/len(prediction))
