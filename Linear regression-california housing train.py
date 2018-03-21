# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 01:33:46 2018
加利福尼亚房屋数据-莫烦的方法
@author: Luna
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

#导入样本数据
train = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
train=train.sample(n=1000)
train['median_house_value']/=1000

learning_rate = 2
training_epochs = 1000
display_step = 50

train_X=train['total_rooms'].values.reshape(-1,1)
train_Y=train["median_house_value"].values.reshape(-1,1)
plt.scatter(train_X, train_Y )

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 定义模型参数
W = tf.Variable(np.random.randn(),name='weights',dtype = (tf.float32))
b = tf.Variable(np.random.randn(),name='bias',dtype = (tf.float32))

pred = tf.add(tf.multiply(W,X),b)
# 定义损失函数
cost = tf.reduce_mean(tf.square(pred-train_Y))
# 使用Adam算法
optimizer = tf.train.AdamOptimizer(learning_rate)
solve = optimizer.minimize(cost)
# 初始化所有变量
init = tf.global_variables_initializer()

# 训练开始
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):        
            sess.run(solve, feed_dict={X:x,Y:y})
        if (epoch+1)% display_step   ==0:
            training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.3f}".format(training_cost), "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization Finished!")
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
        
    plt.plot(train_X, train_Y, 'ro', label="Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()

