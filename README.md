# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vanisha Ramesh
RegisterNumber:  212222040174
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
Initial data set

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104009/953533cb-7ae3-428a-a834-ee66ab765e90)

Data Info

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104009/cd16bb44-eacb-4bbf-aa77-9cb2439796d6)

Optimization of null values

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104009/76335fe7-a3b2-4097-90f4-8269e99c56fc)

Assignment of X and Y values

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104009/8a906fb9-7645-4e42-8901-ade341a0f4bb)

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104009/3456aeab-8353-4e52-a8f8-0eea544f83cf)

Converting string literals to numerical values using label encoder


![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104009/cee8f593-5c52-46b0-a6c8-afd4733153a4)

Accuracy

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104009/f9dace8a-1306-4075-a608-911f047a4bba)

Prediction

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104009/f1c1363c-a349-453a-ba1c-58b57100a611)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
