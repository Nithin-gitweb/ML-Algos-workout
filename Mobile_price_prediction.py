import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#Data_analytics
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
print(df_train.info())
sb.barplot(x = 'clock_speed',y = 'ram', data=df_train)
plt.show()
sb.jointplot(x = 'ram',y = 'price_range', data=df_train,color='Maroon')
plt.show()
sb.boxplot(x = 'price_range',y = 'battery_power', data=df_train)
plt.show()
sb.barplot(x = 'Frontcam', y='Rearcam',hue='price_range',data=df_train)
plt.show()
sb.pairplot(df_train)
plt.show()
#Machine Learning
X = df_train.drop('price_range',axis=1)
y = df_train['price_range']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=64)

#Linear Regression
lr1 = LinearRegression()
lr1.fit(X_train,y_train)
pred = lr1.predict(X_test)
plt.scatter(y_test,pred)
plt.show()

#Logistic Regression
lr2 = LogisticRegression()
lr2.fit(X_train,y_train)
pred2 = lr2.predict(X_test)
print(classification_report(y_test,pred2))

#KNN
arr = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred3 = knn.predict(X_test)
    arr.append(np.mean(pred3 != y_test))
plt.figure(figsize=(12,5))
plt.plot(range(1,50),arr,color = 'red',linestyle = 'dashed',marker = 'o',markerfacecolor = 'blue')
plt.show()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred4 = knn.predict(X_test)
print(classification_report(y_test,pred4))









