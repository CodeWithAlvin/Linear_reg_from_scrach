import linearRegression

x = [x for x in range(10000)]
y = [-600*y+50 for y in range(10000)]

model = linearRegression.LinearRegression()
model.build(input_shape=1,learning_rate=2e-9,epoch=5)
model.fit(x,y)

t_x =  [x for x in range(10000,10020)]
t_y = [-600*y+50 for y in range(10000,10020)]

prediction = model.predict(t_x).flatten()

print(prediction,t_y)

print('loss = ',sum(abs(t_y-prediction))/len(prediction))
