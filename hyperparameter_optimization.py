# 训练集中，随机挑选了一定比例作为验证集，测试集不动作为后续指标测试
# 图片的文件夹需固定为现有状态才能读取图片进行训练和测试，直接运行.py文件即可，文件路径均需保持不变
 

from keras.optimizers import Adam, RMSprop, SGD, Adagrad
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


# 超参数调整
batch_size = 20
epochs_num = 40
train_step_num = 800
val_step_num = 100

# 设置lr随机实验次数
optimization_trial = 50

train_acc_list = []
val_acc_list = []

# 数据集加载
train_generator, validation_generator, test_generator = build_datasets(size, batch_size)

results_val = {}
results_train = {}


for i in range(optimization_trial):
    # lr随机范围
    lr_test = 10 ** np.random.uniform(-6, -1)  # 随机选取lr，范围为1e-6到1e-1

    print("进行到第 %d 次测试" % (i + 1))

    model = build_model(input_shape=(size, size, 3), classes=class_number)
    model.compile(optimizer=RMSprop(lr=lr_test), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(train_generator, 
                              steps_per_epoch=int(train_step_num/batch_size), 
                              epochs=epochs_num,
                              validation_data=validation_generator, 
                              validation_steps=int(val_step_num/batch_size))

    train_acc_list = history.history['accuracy']
    val_acc_list = history.history['val_accuracy']
    val_loss, val_acc = model.evaluate_generator(validation_generator)

    key = "lr:" + str(lr_test)
    print(key + "val_data acc:", val_acc)

    results_val[key] = val_acc_list
    results_train[key] = train_acc_list    


print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 9
col_num = 3
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % col_num: 
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1
    if i >= graph_draw_num:
        break

plt.show()
plt.savefig('hyperparameter choose.png')


