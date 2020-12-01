# medical_images 机器学习大作业

## utils.py
包含数据集建模（build_datasets函数）和模型部分（build_model函数)。
## diseaseClassification_2class.py
为可运行的实验程序。


运行一次大概15-20分钟。
## hyperparameter_optimization.py
为学习率learning_rate的预实验程序，方案为随机选取50个E-06到E-01的lr进行训练，最终print输出val_acc最高的九组结果和对比图(hyperparameter_choose.png)。


当前按照主实验的参数设定运行，实验时间很长，需要十个多小时。
## optimizer_compare.py
为optimizer的预实验程序，在lr固定的情况下，进行SGD、Adam、RMSprop、Adagrad四组实验，print输出形式与总实验相同，但没有画acc和loss图，会输出一张四种optimizer的loss对比图(optimizer_compare.png)。


运行一次大概一小时。
## tips
1. 需要注意每个.py文件中的history.history['accuracy']和 history.history['val_accuracy']，受Keras版本不同，或需修改为acc和val_acc。


2. 已有的对比实验的acc和loss图，序号与excel中对应。
