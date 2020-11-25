#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: 
# @Version: 1.0
# @Author: 任洁
#  @Time    : 2020/11/25 10:06
# @LastEditors: 任洁
#  @File    : utils.py.py
#  @Software: PyCharm


from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models


train_dir = './TrainSet'
validation_dir = './ValidationSet'
test_dir = './TestSet'


def build_datasets(size, batch_size):
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

    return train_generator, validation_generator, test_generator


def build_model(include_top=True, input_shape=(200, 200, 3), classes=2):
    img_input = layers.Input(shape=input_shape)
    x = layers.Convolution2D(32, 5, activation='relu', name='block1_conv1')(img_input)
    x = layers.Convolution2D(32, 5, activation='relu', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='block1_pool')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Convolution2D(64, 3, activation='relu', name='block2_conv1')(x)
    x = layers.Convolution2D(64, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='block2_pool')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Convolution2D(64, 5, activation='relu', name='block3_conv1')(x)
    x = layers.Convolution2D(64, 5, activation='relu', name='block3_conv2')(x)
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