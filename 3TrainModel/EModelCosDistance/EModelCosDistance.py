import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dot, Lambda, Reshape, AveragePooling1D, Dropout, Activation
import keras.optimizers
from keras import backend as K
from keras.models import load_model
from keras.applications import vgg16
import math
import os

#模型的基本参数
input_shape = (118,118,3)
img_nrows = 118
img_ncols = 118
img_channel = 3
fc2_length = 2048
feature_length = 512
class_num = 43
drop_rate = 0.5

#构建VGG16，输出倒数第二全连接层的特征向量，结构为（m,2048）
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    model = vgg16.VGG16(input_tensor=input, weights='imagenet', include_top=False)
    x = Flatten(name='flatten')(model.output)
    x = Dense(fc2_length, activation='relu', name='fc1')(x)
    x = Dropout(drop_rate)(x)
    x = Dense(fc2_length, activation='relu', name='fc2')(x)
    x = Dropout(drop_rate)(x)
    return Model(input, x)

def cosine_distance(input_list):
    input1 = input_list[0]
    input2 = input_list[1]
    norm1 = Dot(axes = 1)([input1,input1])
    norm2 = Dot(axes = 1)([input2,input2])
    dot = Dot(axes = 1)([input1,input2])
    temp = [norm1,norm2,dot]
    temp = Lambda(lambda x:1-temp[2]/(K.sqrt(temp[0])*K.sqrt(temp[1])))(temp)
    return temp

def EModelCosDistance(weights='imagenet', input_shape=input_shape, classes=class_num, precondition_length=4, effect_length=4):
    #多个视频帧作为输入
    input_tensor_list = list()
    for i in range(precondition_length + effect_length):
        input_tensor_list.append(Input(input_shape))
    #Precondition
    top_list = list()
    top_network = create_base_network(input_shape)
    for i in range(precondition_length):
        top_list.append(top_network(input_tensor_list[i]))
    for i in range(len(top_list)):
        top_list[i] = Reshape((1, fc2_length), input_shape=(fc2_length,))(top_list[i])
    precondition_tensor = Lambda(lambda x:K.concatenate([x[i] for i in range(len(x))],axis=1))(top_list)
    print("Precondition Before Average:",precondition_tensor.shape)
    precondition_tensor = AveragePooling1D(precondition_length)(precondition_tensor)
    print("Precondition After Average:",precondition_tensor.shape)
    precondition_tensor = Reshape((fc2_length,), input_shape=(1,fc2_length))(precondition_tensor)
    print("Precondition After Averagere reshape:",precondition_tensor.shape)
    #此处全连接层的激活函数随后调整
    precondition_tensor = Dense(feature_length, activation="relu")(precondition_tensor)

    #Effect
    down_list = list()
    down_network = create_base_network(input_shape)
    for i in range(precondition_length,precondition_length + effect_length):
        down_list.append(down_network(input_tensor_list[i]))
    for i in range(len(down_list)):
        down_list[i] = Reshape((1, fc2_length), input_shape=(fc2_length,))(down_list[i])
    effect_tensor = Lambda(lambda x:K.concatenate([x[i] for i in range(len(x))],axis=1))(down_list)
    print("Effct Before Average:",effect_tensor.shape)
    effect_tensor = AveragePooling1D(effect_length)(effect_tensor)
    print("Effect After Average:",effect_tensor.shape)
    effect_tensor = Reshape((fc2_length,), input_shape=(1,fc2_length))(effect_tensor)
    print("Effect After Averagere reshape:",effect_tensor.shape)
    #此处全连接层的激活函数随后调整
    effect_tensor = Dense(feature_length, activation="relu")(effect_tensor)

    #将precondition_tensor放入转移矩阵并计算与effect_tensor的1-cosine
    precondition_tensors = list()
    for i in range(class_num):
        precondition_tensors.append(Dense(feature_length,use_bias = False)(precondition_tensor))
    result_tensors = list()
    for i in range(class_num):
        result_tensors.append(cosine_distance([precondition_tensors[i],effect_tensor]))
    result_tensor = Lambda(lambda x:K.concatenate([x[i] for i in range(len(x))],axis=1))(result_tensors)
    model = Model(input_tensor_list, result_tensor, name='ActionRecognition')
    return model
