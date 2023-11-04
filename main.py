import numpy as np
from data import *
from loss_fn import *
from Layer import *

from activation import *
np.random.seed(0) #So that data won't change

dense1 = Layer_Dense(2,3)
activation1=Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense3=Layer_Dense(3,3)
dense4=Layer_Dense(3,3)

X,y = spiral_data(100,3) #input
dense1.forward(X)
activation1.forward(dense1.output)
print(activation1.output[:5]) #Shows top 5


dense2.forward(activation1.output)
print(dense2.output[:5])
activation1.forward(dense2.output)

dense3.forward(activation1.output)
activation1.forward(dense3.output)

dense4.forward(activation1.output)
activation2.forward(dense4.output)



print("FINAL OUTPUT : ")
print(activation2.output[:5])
Loss_function = Loss_CategoricalCrossenrophy()
l = Loss_function.calculate(activation2.output,y)
print("Loss : ", l)

