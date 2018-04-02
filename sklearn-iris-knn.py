import numpy as np
from sklearn import datasets
# 导入数据
iris = datasets.load_iris()
# 观察数据结构
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

print(type(iris.data))
print(type(iris.target))

print(iris.data.shape)
print(iris.target.shape)


#将整个数据集拆分为两部分：训练集和测试集
X = iris.data
y = iris.target
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
#配置Knn分类算法：K=5
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

#训练
knn.fit(X_train, y_train)

#测试
y_pred = knn.predict(X_test)

#评估
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

#k取1-25时分类的准确率展示
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
