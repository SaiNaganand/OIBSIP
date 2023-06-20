import pandas as pd
import numpy as np
data=pd.read_csv("/content/iris.data")
print(data)
x=data.iloc[:,0:4]
y=data.iloc[:,4]
print(y)
print(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test)*100)
