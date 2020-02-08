import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

plt.imshow(x_train[0])
plt.show()
# 2次元データを数値に変換
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train[0])


plt.imshow(x_train[0], cmap='gray')
plt.show()
