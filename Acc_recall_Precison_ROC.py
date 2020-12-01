#!/usr/bin/env python
# coding=utf-8

# @Description  : 
# @Version      : 1.0
# @Author       : 任洁
# @Date         : 2020-11-23 15:06:41
# @LastEditors  : 任洁
# @LastEditTime : 2020-12-01 19:32:08
# @FilePath     : /Desktop/medical_images/Acc_recall_Precison_ROC.py



from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score  # 计算各种指标的函数

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model 
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model = load_model('dn_2class40epoch1201.h5')


def onehot_labels(labels):  # One-hot encoding the label
    return np.eye(2)[labels]


class_number = 2
size = 200
batch_size = 20
test_dir = './TestSet'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator=test_datagen.flow_from_directory(
            test_dir,
            target_size=(size,size),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode="rgb",
            shuffle=False  # 很重要，按照文件顺序读取，保证标签与预测对应是同一张图片
            )

Y_pred = model.predict_generator(test_generator)  # Y_pred为数据的预测值(预测的标签)
Y_train = onehot_labels(test_generator.classes)  # Y_train 为数据的真实标签，这里是one hot 格式一个样本对应1*2的标签

#############################
Y_pred_0 = [y[1] for y in Y_pred]  # 取出y中的一列
Y_train0 = [y[1] for y in Y_train]


# Y_train0为真实标签，Y_pred_0为预测标签，这里roc_curve函数需要是为一维的输入
fpr, tpr, thresholds_keras = roc_curve(Y_train0, Y_pred_0)   
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
auc_area = auc(fpr, tpr)  # 计算auc面积
plt.plot(fpr, tpr, label='(AUC area = {:.3f})'.format(auc_area))

# plt.xlabel('Sensitivity')#等价
# plt.ylabel('1-Specificity')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('ROC1201.jpg')
plt.show()    

print("AUC area : ", auc_area)
# print("AUC area : ",roc_auc_score(Y_train0, Y_pred_0))

accuracy = accuracy_score(Y_train0, [round(i) for i in Y_pred_0])
recall = recall_score(Y_train0, [round(i) for i in Y_pred_0])
precision = precision_score(Y_train0, [round(i) for i in Y_pred_0])
f1_score = f1_score(Y_train0, [round(i) for i in Y_pred_0])

print('accuracy:', accuracy)
print('recall:', recall)
print('precision:', precision)
print('f1_score:', f1_score)
