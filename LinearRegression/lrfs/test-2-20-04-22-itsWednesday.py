import linearRegression

x =  [x for x in range(1000,1020)]
y = [y**2 - 2*y for y in x ]

model = linearRegression.LinearRegression()
model.build(input_shape=2,learning_rate=1e-7,epoch=10)
model.fit(x,y)

t_x =  [x for x in range(1000,1020)]
t_y = [y**2 -2*y for y in t_x]

prediction = model.predict(t_x).flatten()

print(prediction,t_y)

print('loss = ',sum(abs(t_y-prediction))/len(prediction))
