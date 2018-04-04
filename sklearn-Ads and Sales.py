import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

df = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)


print(df.head())
sns.pairplot(df,x_vars=['TV','radio','newspaper'],y_vars='sales',size=8,aspect=0.9,kind='reg')

X=df[['TV','radio','newspaper']]
y=df['sales']
print(type(X))
print(type(y))


X_train,X_test,y_train,y_test = train_test_split( X, y,random_state=1)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
linreg = LinearRegression()
linreg.fit(X_train,y_train)

LinearRegression(copy_X=True,fit_intercept=True,normalize=False,n_jobs=1)
print(linreg.intercept_)
print(linreg.coef_)

y_pred = linreg.predict(X_test)
print(y_pred)
print ("MAE:",metrics.mean_absolute_error(y_test, y_pred))
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

feature_cols=['TV','radio']
X = df[feature_cols]
y = df['sales']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
linreg = LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)
print(linreg.coef_)
y_pred = linreg.predict(X_test)
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))