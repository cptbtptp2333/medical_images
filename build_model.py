#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: 
# @Version: 1.0
# @Author: 任洁
#  @Time    : 2020/11/25 10:03
# @LastEditors: 任洁
#  @File    : build_model.py.py
#  @Software: PyCharm


from keras import layers
from keras import models


def build_model(include_top=True, input_shape=(size, size, 3), classes=class_number):
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

