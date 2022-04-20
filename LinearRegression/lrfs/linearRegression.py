import  numpy as np # matrix operation
import pickle
'''
input shape = n

equation => 
w1.x1 + w2.x2 + ..... + wn.xn = h(x)

loss = MSE   J(w) = 1/m â‚¬ [h(x) - y ]^2
d(J(w))/dw = -1/m (y-h(x)) 2x



optimizer GD minimize loss with respect to every Wi

w := w - Ã¦ * d(J(w))/dw


run iteration for x repetation

# HOW TO RUN
model = LinearRegression()
model.build(input_shape=1,learning_rate=4e-3)
model.fit(X,y)
model.predict()
'''
class LinearRegression:
    def __init__(self):
        pass
	
    def loss(self):
        whole_mse = sum( (self.y - self.predict(self.x))**2 )/len(self.x)
        return whole_mse
    
    def build(self,input_shape,learning_rate,batch_size=32,epoch=5):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.input_shape = input_shape
        self.weights = np.random.random(self.input_shape)
        self.bias = np.ones(1)
		
    def fit(self,x,y):
        self.x = np.array(x).reshape(-1,self.input_shape)
        self.y = np.array(y).reshape(len(self.x),-1)
        self.learn()
	
    def optimizer(self,):
        #implement all
        '''
        w := w - Ã¦. d(J)/dw
        d(J(w))/dw = -2/m (y-h(x)) x
        '''
        steps = np.floor(len(self.x)/self.batch_size)
        for _ in range(int(steps)):
            random_index = np.random.randint(0,len(self.x),self.batch_size)
            predictions = self.predict(self.x[random_index])
            diffrence = self.y[random_index] - predictions
            gradients = np.sum(2 * diffrence*self.x[random_index])/self.batch_size
            self.weights = self.weights + (self.learning_rate * gradients)
            self.bias = self.bias + ((self.learning_rate * np.sum(2 * diffrence)) /self.batch_size)
		
	
    def learn(self):
        for i in range(1,self.epoch+1):
            self.optimizer()
            loss = self.loss()
            self.print_info(i,loss)
	
    def print_info(self,i,loss):
        print(f'\n[{i}/{self.epoch}] epoch  - loss : {loss}\n')
		
    def predict(self,X):
        return  np.array(list(map(lambda x: sum(x*self.weights + self.bias) , X))).reshape(len(X),-1)
	
    def save(self,path):
        mdl = {'weights':self.weights,
                    'bias':self.bias,
        			'lr':self.learning_rate,
        }
        with open(path,"wb") as f:
            pickle.dump(mdl,f)
	
    def load(self,path):
        with open(path,"rb") as f:
            self.weights, self.bias, self.learning_rate = pickle.load(f).values()