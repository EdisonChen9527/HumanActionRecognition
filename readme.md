#label文件夹
* trainlist.txt：训练数据（帧文件路径、分类标注），共计7260个视频；
* testlist.txt：测试数据（帧文件路径、分类标注），共计3974个视频；
* classlist.txt：活动类别及其对应编号。
#1Data_division
* 处理训练数据和测试数据，记录数据（路径，帧数，帧list，类别），将其存储为python的list，element为上述元组；
* main.py:输入label文件夹内训练/测试数据的文档和ACT数据库的位置，生成相应的python list；
* train_list.txt: main.py处理训练集的结果；
* test_list.txt： main.py处理测试集的结果；
* python ./1Data_division/main.py path\_to\_ACT data\_record
* python ./1Data\_division/main.py ~/DL\_project/ACT/action\_frames\_release ./label/trainlist.txt
#2Data_process
* 检测每个视频是否足以提取25帧；对能够提取25帧的视频进行resample；
* DataAnalysis.py：用于计算视频中不足25帧的比率，得出结论（无法简单地忽略不足25帧的视频，比例较大，需要加以处理）；
* main.py：对数据进行重采样和矩阵化（重采样的策略有待调整，是否应该重视最后一帧）；
* python 2Data_process/main.py ~/DL\_project/ACT/action\_frames\_release 1Data\_division/test\_list.txt ./test
* TestNpyMatrix.py：用于检查生成的npy矩阵是否正确；
* python 2Data\_process/TestNpyMatrix.py ./train 100
* AddTrainFromTest.py：从测试集中拿出一部分数据添加到训练集，以降低variance；
#train
* input和Y文件夹分别存放训练集矩阵化后的npy数据；
#test
* input和Y文件夹分别存放测试集矩阵化后的npy数据；
#picture
* 用于存放测试用的视频帧图片。
#3TrainModel
##完成模型的测试代码和训练；
##EModel
* EModel.py：存放最简单的模型，由main.py进行调用；
* EModelLarger.py：存放了在EModel基础上，扩大了参数规模，并将优化算法由Adam改为了SGD；
* main.py：使用训练数据对模型进行训练（注意将训练数据打乱进行训练）；
* python ./3TrainModel/main.py ./train
##EModelCosDistance
* EModelCosDistance.py：存放EModelCosDistance模型，按照论文实现损失函数；
* mainForCos.py：使用EModelCosDistance进行训练，并且包含测试任务，原因是Model含有Lambda Layer难以存储；
* python ./3TrainModel/mainForCos.py ./train
##EModelXception
* EModelXception.py
* mainEModelXception.py
#4TestModel
* 对训练完保存下来的模型进行测试；
#Model
* 存放训练完成的模型；
* mymodel.h5：截取4对图片，[6,7,8,9]和[18,19,20,21]训练而成；优化算法是Adam，learning\_rate为0.01，训练次数为50轮；
#TestTensorFunction
* 存放tensor计算函数的测试代码；
* TestCosDistance.py
#Nohup
* 用于存放各个nohup执行得出的结果文件；
* nohupFor1[6,7,8,9,18,19,20,21].out：仅训练1轮，用于充当测试的代码，47%准确率；
* nohupFor4[6,7,8,9,18,19,20,21].out：经过训练4轮，56%准确率；
* nohupFor4[0,1,2,3,21,22,23,24].out：经过训练4轮，57%准确率；
* nohupFor5[6,7,8,9,18,19,20,21].out：经过训练5轮，显存用尽；
* nohupFor4[0,1,23,24].out：经过训练4轮，56%准确率；
* nohupForTrain.out：模型对于训练集的准确率，90.9%准确率；
* nohupFor5[0,1,23,24].out：经过训练5轮，57.5%准确率；
* nohupFor4[0,1,2,3,12,12,21,22,23,24].out：显存直接溢出；
#6Visualization
* 包含模型结果的可视化
* confusionMatrixData.txt：Confusion Matrix的数据来源；