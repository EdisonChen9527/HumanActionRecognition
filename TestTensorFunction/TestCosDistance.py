import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dot, Lambda, Reshape, AveragePooling1D, Dropout
from keras import backend as K

def cosine_distance(input_list):
    input1 = input_list[0]
    input2 = input_list[1]
    norm1 = Dot(axes = 1)([input1,input1])
    norm2 = Dot(axes = 1)([input2,input2])
    dot = Dot(axes = 1)([input1,input2])
    temp = [norm1,norm2,dot]
    temp = Lambda(lambda x:1-temp[2]/(K.sqrt(temp[0])*K.sqrt(temp[1])))(temp)
    return temp

x1 = Input(shape=(4,))
x2 = Input(shape=(4,))
Y = cosine_distance([x1,x2])
testModel = Model([x1,x2], Y, name='TestCosDistance')
testModel.summary()
