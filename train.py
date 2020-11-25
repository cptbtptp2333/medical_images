#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: 保存和加载模型刚加上，没有经过调试，可能有bug
# @Version: 1.0
# @Author: 任洁
#  @Time    : 2020/11/9 13:12
# @LastEditors: 任洁
#  @File    : train.py.py
#  @Software: PyCharm


import tensorflow as tf
from load_data import load_data
from pathlib import Path
from tensorflow.keras import Model, layers
import matplotlib.pyplot as plt


num_classes = 2

learning_rate = 0.001
training_step = 100
display_step = 5
batch_size = 16

conv_filter1 = 32
conv_filter2 = 64
fcl_units = 1024
dropout_rate = 0.5


# path_dir为当前目录
path_dir = Path.cwd()
train_path = Path(path_dir/"2-MedImage-TrainSet")
test_path = Path(path_dir/"2-MedImage-TestSet")

train_ds, train_num = load_data(train_path)
test_ds, test_num = load_data(test_path)

# train_data = train_ds.repeat().shuffle(buffer_size=train_num).batch(batch_size).prefetch(1)
# test_data = test_ds.repeat().shuffle(buffer_size=test_num).batch(batch_size).prefetch(1)

train_data = train_ds.repeat().batch(batch_size).prefetch(1)
test_data = test_ds.repeat().batch(batch_size).prefetch(1)

# (test_image, test_label) = test_data  # FIXME





convnet = ConvNet()
convnet.load_weights(filepath="./tfmodel.ckpt")


def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(
        tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


optimizer = tf.optimizers.Adam(learning_rate)


def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = convnet(x, is_training=True)
        loss = cross_entropy_loss(pred, y)

    trainable_variables = convnet.trainable_variables
    gradients = g.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))


for step, (batch_img, batch_label) in enumerate(train_data.take(training_step), 1):
    run_optimization(batch_img, batch_label)

    # 保存模型
    convnet.save_weights(filepath="./tfmodel.ckpt")

    if step % display_step == 0:
        pred = convnet(batch_img)
        loss = cross_entropy_loss(pred, batch_label)
        acc = accuracy(pred, batch_label)
        print('step = %i, loss = %f, accuracy = %f' % (step, loss, acc))

# pred = convnet(test_image)
# print('Test accuracy is %.3f%%' % (100 * accuracy(pred, test_label)))
