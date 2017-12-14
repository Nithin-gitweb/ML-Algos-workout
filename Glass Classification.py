import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('glass.csv')
df = pd.DataFrame(df)
print(df.head(),df.info(),'\n',df['Type'].unique())
sb.boxplot(x = 'Type',y= 'RI',data=df,color='red')
plt.show()
X = df.drop('RI',axis=1)
y = df['RI']
X_Train,X_test,y_train,y_test = train_test_split(X,y)
model = LinearRegression()
model.fit(X_Train,y_train)
pred = model.predict(X_test)
plt.scatter(x=y_test,y=pred,cmap='coolwarm')
plt.show()
model2 = LogisticRegression()
X2 = df.drop('Type',axis=1)
y2 = df['Type']
X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y2)
model2.fit(X2_train,y2_train)
pred2 = model2.predict(X2_test)
print(classification_report(y2_test,pred2))






