# 训练集中，随机挑选了一定比例作为验证集，测试集不动作为后续指标测试
# 图片的文件夹需固定为现有状态才能读取图片进行训练和测试，直接运行.py文件即可，文件路径均需保持不变

from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adagrad, RMSprop, SGD

from keras.utils import plot_model
from keras.models import load_model 

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


class_number = 2
size = 200

def build_model(include_top=True, input_shape=(size, size, 3), classes=class_number):
    img_input = layers.Input(shape=input_shape)
    x = layers.Convolution2D(32, 5,  activation='relu',  name='block1_conv1')(img_input)
    x = layers.Convolution2D(32, 5,  activation='relu',  name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2),  name='block1_pool')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Convolution2D(64, 3, activation='relu',  name='block2_conv1')(x)
    x = layers.Convolution2D(64, 3,  activation='relu', border_mode='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2),  name='block2_pool')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Convolution2D(64, 5,  activation='relu',  name='block3_conv1')(x)
    x = layers.Convolution2D(64, 5, activation='relu',  name='block3_conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='block3_pool')(x)
    x = layers.BatchNormalization()(x)

    if include_top:
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(20, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
 
    model = models.Model(img_input, x, name='model')
    model.summary()
    return model


train_dir = './TrainSet'
validation_dir = './ValidationSet'
test_dir = './TestSet'

# 参数设置
batch_size = 20
learning_rate = 1e-2
epochs_num = 40
train_step_num = 800
val_step_num = 100

train_acc_list = []
val_acc_list = []


# datagenerator 部分
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                   width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=False)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(size,size), 
            batch_size=batch_size,
            class_mode='categorical',
            color_mode="rgb",
            )

validation_generator = valid_datagen.flow_from_directory(
            validation_dir,
            target_size=(size, size),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode="rgb",
            )

test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(size, size),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode="rgb",
            shuffle=False
            )


optimizers = {}
loss = {}
optimizers['SGD'] = SGD(lr=learning_rate)
optimizers['Adam'] = Adam(lr=learning_rate)
optimizers['RMSprop'] = RMSprop(lr=learning_rate)
optimizers['Adagrad'] = Adagrad(lr=learning_rate)

for key in optimizers:
    model = build_model()
    model.compile(optimizer=optimizers[key], loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(train_generator, 
                              steps_per_epoch=int(train_step_num/batch_size), 
                              epochs=epochs_num,
                              validation_data=validation_generator, 
                              validation_steps=int(val_step_num/batch_size))
    loss[key] = history.history['loss']
    print("optimizer:", key + "train_data loss:", loss[key])
    print('traindata loss and acc:', model.evaluate_generator(train_generator, steps=4))
    print('validation data loss and acc:', model.evaluate_generator(validation_generator, steps=4))
    print('testdata loss and acc :', model.evaluate_generator(test_generator))


markers = {"SGD": "o", "Adam": "D", "RMSprop": "x", "Adagrad": "s"}
x = np.arange(epochs_num)
for key in optimizers.keys():
    plt.plot(x, loss[key], marker=markers[key], label=key)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()

plt.show()
plt.savefig('optimizer compare.png')
