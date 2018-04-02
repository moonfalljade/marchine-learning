
from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 读入数据
train = pd.read_csv("sample_submission.csv")
train=train.sample(n=1000)
# 选取房屋面积小于１２０００的数据
train = train[train['LotArea'] < 12000] 
train_X = train['LotArea'].values.reshape(-1, 1)
train_Y = train['SalePrice'].values.reshape(-1, 1)

n_samples = train_X.shape[0]
# 学习率
learning_rate = 3
# 迭代次数
training_epochs = 1000
# 每多少次输出一次迭代结果
display_step = 50

# 这个X和Y和上面的train_X,train_Y是不一样的，这里只是个占位符，它是个不断被被赋值的变量
# 训练开始的时候需要“喂”(feed)数据给它
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 定义模型参数
# numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。
W = tf.Variable(np.random.randn(), name="weight", dtype=tf.float32)
b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)

# 定义模型
pred = tf.add(tf.multiply(W, X), b)
# 定义损失函数
cost = tf.reduce_sum(tf.pow(pred-Y, 2)) / (2 * n_samples)
# 使用Adam算法
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

# 训练开始
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if (epoch + 1) % display_step == 0:
            training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.3f}".format(training_cost), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # 画图
    plt.plot(train_X, train_Y, 'ro', label="Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()
