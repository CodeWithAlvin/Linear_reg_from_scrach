import linearRegression

x = [[x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x] for x in range(10000)]
y = [48*y+1 for y in range(10000)]

model = linearRegression.LinearRegression()
model.build(input_shape=24,learning_rate=1e-11,epoch=50)
model.fit(x,y)

t_x =  [[x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x] for x in range(10000,10020)]
t_y = [48*y+1 for y in range(10000,10020)]

prediction = model.predict(t_x).flatten()

print(prediction,t_y)

print('loss = ',sum(abs(t_y-prediction))/len(prediction))
