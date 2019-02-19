pip install tensorflow keras

%matplotlib inline
from __future__ import division, print_function

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Kerasに付属の手書き数字画像データをダウンロード
np.random.seed(0)
(X_train, labels_train), (X_test, labels_test) = mnist.load_data()

# Traning setのラベルを確認
labels_train

# Training setの概要確認
print(X_train.shape, labels_train.shape, sep='¥n')

# Test setの概要確認
print(X_test.shape, labels_test.shape, sep='¥n')

# Training setの画像を表示

import matplotlib.pyplot as plt

label_images = {label: [] for label in set(labels_train.tolist())}
for x, label in zip(X_train, labels_train):
    if all(len(images) >= 10 for images in label_images.values()):
        break
    if len(label_images[label]) >= 10:
        continue
    label_images[label].append(x)

for j, (label, images) in enumerate(label_images.items()):
    plt.subplot(10, 11, j * 11 + 1)
    plt.text(0.5, 0.5, label, ha='center', va='center')
    plt.axis('off')
    for i, image in enumerate(images):
        if i >= 10:
            continue
        plt.subplot(10, 11, j * 11 +  i + 2)
        plt.imshow(image, cmap='Greys_r')
        plt.axis('off')
plt.show()

del label_images

# 各画像は行列なのでベクトルに変換→X_trainとX_testを作成
X_train=X_train.reshape(len(X_train),-1)
X_test=X_test.reshape(len(X_test),-1)

# ラベルをone-hotベクトル（値がひとつだけ1で他が0のベクトル）に変換→Y_trainとY_testを作成
Y_train = to_categorical(labels_train)
Y_test = to_categorical(labels_test)

# モデルの準備 単純パーセプロン→多層パーセプロン
model = Sequential()
model.add(Dense(10, input_shape=(784,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(20))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Trainingの実施
model.fit(X_train, Y_train, batch_size=128, nb_epoch=10, verbose=1)

# Test dataを用いてモデルを評価
_, acc = model.evaluate(X_test, Y_test, verbose=0)
print('accuracy: {}'.format(acc))

# Classification_report関数で評価結果を表示
labels_pred = model.predict_classes(X_test, verbose=0)
print(confusion_matrix(labels_test, labels_pred))
print(classification_report(labels_test, labels_pred))
