#!/usr/bin/env python
# coding=utf-8

# @Description  : history.history['accuracy']，版本不同，或为['acc']，val_acc同理
# @Version      : 2.0
# @Author       : 任洁
# @Date         : 2020-11-25 10:41:36
# @LastEditors  : 任洁
# @LastEditTime : 2020-12-01 19:05:45
# @FilePath     : /Desktop/medical_images/diseaseClassification_2class.py

# 训练集中，随机挑选了一定比例作为验证集，测试集不动作为后续指标测试
# 图片的文件夹需固定为现有状态才能读取图片进行训练和测试，直接运行.py文件即可，文件路径均需保持不变


from keras.optimizers import Adam, RMSprop
from utils import build_model, build_datasets

from keras.utils import plot_model
from keras.models import load_model 


import matplotlib.pyplot as plt
import os
import warnings
warnings.simplefilter('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class_number = 2
size = 200


# 超参数调整
batch_size = 20
learning_rate = 1e-5
epochs_num = 40
train_step_num = 800
val_step_num = 100
optimizer_now = Adam(lr=learning_rate)


# DataGenerator
train_generator, validation_generator, test_generator = build_datasets(size, batch_size)


# build_model
model = build_model(input_shape=(size, size, 3), classes=class_number)

model.compile(optimizer=optimizer_now, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_generator, 
                              steps_per_epoch=int(train_step_num/batch_size), 
                              epochs=epochs_num,
                              validation_data=validation_generator, 
                              validation_steps=int(val_step_num/batch_size))


print('traindata loss and acc:', model.evaluate_generator(train_generator, steps=4))
print('validation data loss and acc:', model.evaluate_generator(validation_generator, steps=4))
print('testdata loss and acc :', model.evaluate_generator(test_generator))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


plt.figure(1)
# 画出训练集和验证集的acc变化图
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('TrainingandvalidationAccuracy7.jpg')


plt.figure(2)
# 画出训练集和验证集的loss变化图
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
plt.savefig('TrainingandvalidationLoss7.jpg')


# 保存模型，保存模型权重
model.save('dn_2class40epoch1201.h5')
model.save_weights('dn_2class__weights40epoch1201.h5')
