import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

data_source = "./6Visualization/confusionMatrixData.txt"
lines = open(data_source,"r").readlines()
class_num = 43

labels = list()
predicts = list()
for line in lines:
    if "Example" in line:
        labels.append(int(line.split()[len(line.split())-1]))
    if "Predict" in line:
        predicts.append(int(line.split()[len(line.split())-1]))

confusion_matrix = np.zeros((class_num,class_num))
for i in range(len(labels)):
    label = labels[i]
    predict = predicts[i]
    confusion_matrix[label-1,predict-1] += 1
print(confusion_matrix)

df_cm = pd.DataFrame(confusion_matrix, index = [(i+1) for i in range(class_num)], columns = [(i+1) for i in range(class_num)])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrix.png', format='png')
