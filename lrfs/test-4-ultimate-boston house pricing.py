from sklearn import datasets
from sklearn.model_selection import train_test_split
import linearRegression
import numpy as np

data = datasets.load_boston()
x = data.data
y = data.target

x,test_x,y,test_y = train_test_split(x,y,random_state=42,train_size=0.8)


model = linearRegression.LinearRegression()
model.build(input_shape=13,learning_rate=1e-9,epoch=5000)
model.fit(x,y)
model.save('lrfs/boston.mdl')
prediction = model.predict(test_x).flatten()

print(prediction,test_y)

print('loss = ',sum(abs(test_y-prediction))/len(prediction))
