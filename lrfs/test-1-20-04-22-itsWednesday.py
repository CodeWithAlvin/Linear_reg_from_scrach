import linearRegression

x = [x for x in range(10000)]
y = [-600*y+50 for y in range(10000)]

model = linearRegression.LinearRegression()
model.build(input_shape=1,learning_rate=2e-9,epoch=5)
model.fit(x,y)

t_x =  [x for x in range(10000,10020)]
t_y = [-600*y+50 for y in range(10000,10020)]

prediction = model.predict(t_x).flatten()

for i in range(len(t_x)):
    print(f'{t_y[i]} , {prediction[i]:.1f}')

print(f'loss = {sum(abs(t_y-prediction))/len(prediction):.2f}')
