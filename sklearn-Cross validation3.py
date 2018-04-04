from sklearn import datasets
import numpy as np
from sklearn.learning_curve import validation_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X=digits.data
y=digits.target

#通过尝试不同的SVC model的系数gamma来对比training和 testing的loss
param_range = np.logspace(-6,-2.3,5)
train_loss,test_loss = validation_curve(
        SVC(),X,y,param_name='gamma',param_range=param_range,cv=10,
        scoring='mean_squared_error')

train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)

#gamma的值在0.0005-0.0006的范围时候，train和test数据的loss都较小
#但当gamma的值越大，training的loss变小，而test的loss变大。这是因为model overfiting的缘故，导致test的准确度下降
plt.plot(param_range,train_loss_mean,'o-',color='r',label='training')
plt.plot(param_range,test_loss_mean,'o-',color='g',label='Cross-validation')
plt.xlabel('gamma')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()