#!/usr/bin/env python
# coding=utf-8

# @Description  : 
# @Version      : 1.0
# @Author       : 任洁
# @Date         : 2020-11-06 15:11:41
# @LastEditors  : 任洁
# @LastEditTime : 2020-11-06 18:21:34
# @FilePath     : \medical images\load_data.py

import tensorflow as tf
import os
from pathlib import Path
import random
# AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [1500, 1500])  # FIXME resize成什么大小
  image /= 255.0  # normalize to [0,1] range
  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label


def load_data(p):

    """
    input:
    :param p : 数据库路径
    :returns：
    :param image_label_ds :图像-标签对数据库
    :param image_count : 数据库图像数量
    """

    all_image_paths = list(p.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)  # 将数据打乱

    image_count = len(all_image_paths)
    
    label_names = sorted(item.name for item in p.glob('*/') if item.is_dir())

    # 标签：患病为0，正常为1
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[Path(path).parent.name]
                    for path in all_image_paths]

    # DataSet的建立
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    
    return image_label_ds, image_count


# path_dir为当前目录
path_dir = Path.cwd()
p1 = Path(path_dir/"2-MedImage-TrainSet")
p2 = Path(path_dir/"2-MedImage-TestSet")

train_ds, train_num = load_data(p1)
test_ds, test_num = load_data(p2)






