# 训练集中，随机挑选了一定比例作为验证集，测试集不动作为后续指标测试
# 图片的文件夹需固定为现有状态才能读取图片进行训练和测试，直接运行.py文件即可，文件路径均需保持不变


from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from utils import build_model, build_datasets

from keras.utils import plot_model
from keras.models import load_model 


import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.simplefilter('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class_number = 2
size = 200


# 参数设置
batch_size = 20
learning_rate = 1e-2
epochs_num = 40
train_step_num = 800
val_step_num = 100

train_acc_list = []
val_acc_list = []


train_generator, validation_generator, test_generator = build_datasets(size, batch_size)

optimizers = {}
loss = {}
optimizers['SGD'] = SGD(lr=learning_rate)
optimizers['Adam'] = Adam(lr=learning_rate)
optimizers['RMSprop'] = RMSprop(lr=learning_rate)
optimizers['Adagrad'] = Adagrad(lr=learning_rate)

for key in optimizers:
    model = build_model(include_top=True, input_shape=(size, size, 3), classes=class_number)
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
