from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X=iris.data
y=iris.target

k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy') #for classification
    #loss = -cross_val_score(knn,X,y,cv=10,scoring='mean_squared_error') for regression
    k_scores.append(scores.mean())

#k值取的越大越容易underfitting,所以分类准确度下降     
plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

