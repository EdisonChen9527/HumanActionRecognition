import EModelXception
from numpy import random
import argparse
import numpy as np
import math
import keras.optimizers
from keras import backend as K

parser = argparse.ArgumentParser(description='Get data to train')
parser.add_argument('path_to_train', metavar='base', type=str, help='Path to ACT train data.')
args = parser.parse_args()
path_to_train = args.path_to_train

#模型的基本参数
precondition_index_list = [0,1,2,3]
effect_index_list = [21,22,23,24]
index_list = [0,1,2,3,21,22,23,24]
input_length = len(index_list)
epoch_total = 1
train_group_count = 633
test_group_count = 70
class_num = 43
learn_rate = 0.01
img_nrows = 118
img_ncols = 118
img_channel = 3

mymodel = EModelXception.EModelCosDistance(precondition_length=len(precondition_index_list), effect_length=len(effect_index_list))
mymodel.summary()

def contrastive_loss(y_true, y_pred):
    margin = 0.5
    y_true_four = y_true * 4
    loss_true = K.maximum(y_pred - y_true_four, 0)
    loss_false = y_pred - y_true
    loss_false = K.minimum(loss_false, 0)
    loss_false = K.abs(loss_false)
    loss_result = K.sum(loss_true + loss_false)
    return loss_result

my_optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=False)
mymodel.compile(loss=contrastive_loss, optimizer=my_optimizer, metrics=[contrastive_loss])


def create_input(group, precondition_list, effect_list):
    X_list = list()
    Y = np.load(path_to_train + "/Y/" + "Y_g" + str(group).zfill(3) + ".npy")
    for i in range(len(precondition_list)):
        temp = np.load(path_to_train + "/input/" + "input_g" + str(group).zfill(3) + "_" + str(precondition_list[i]).zfill(2) +".npy")
        X_list.append(temp)
    for i in range(len(effect_list)):
        temp = np.load(path_to_train + "/input/" + "input_g" + str(group).zfill(3) + "_" + str(effect_list[i]).zfill(2) +".npy")
        X_list.append(temp)
    return X_list,Y

for i in range(epoch_total):
    for j in range(train_group_count):
        X,Y = create_input(j+1, precondition_index_list, effect_index_list)
        (Y_length,Y_width) = Y.shape
        for m in range(Y_length):
            for n in range(Y_width):
                if Y[m,n] == 0:
                    Y[m,n] = 0.5
                else:
                    Y[m,n] = 0
        print("Recurrent epoch:",str(i+1),"train_group_count:",str(j+1))
        '''
        mymodel.fit(X,Y,batch_size=16,epochs=1)
        '''
        for k in range(Y_length):
            X_temp = list()
            for l in range(len(X)):
                X_temp.append(X[l][k].reshape(1,img_nrows,img_ncols,img_channel))
            mymodel.fit(X_temp,Y[k].reshape(1,class_num),batch_size=1,epochs=1)

'''
#存储带有Lambda Layer的模型容易出现问题，换取nohup.out直接看模型效果
mymodel.save('./Model/EModelCosDistance.h5')
'''

#完成了模型的训练，对其性能进行测试
path_to_train = "./test"
accuracy_rates = list()
for i in range(test_group_count):
    accuracy_rate = 0
    X,Y = create_input(i+1, precondition_index_list, effect_index_list)
    (Y_length,Y_width) = Y.shape
    for m in range(Y_length):
        for n in range(Y_width):
            if Y[m,n] == 0:
                Y[m,n] = 0.5
            else:
                Y[m,n] = 0
    Y_pred = mymodel.predict(X,batch_size=16)
    print("test_group_count:",str(i+1))
    Y_pred = np.argmin(Y_pred, axis=1)
    Y = np.argmin(Y, axis=1)
    for j in range(Y_length):
        print("\tExample",str(j+1),"is class",str(Y[j]+1))
        print("\tPredict",str(j+1),"is class",str(Y_pred[j]+1))
        if Y[j] == Y_pred[j]:
            accuracy_rate += 1
    accuracy_rate = accuracy_rate/Y_length
    accuracy_rates.append(accuracy_rate)
sum = 0
for i in range(len(accuracy_rates)):
    sum += accuracy_rates[i]
print("Average accuracy with",epoch_total,"training:",str(sum/len(accuracy_rates)))
print("accuracy_rates:",accuracy_rates)
