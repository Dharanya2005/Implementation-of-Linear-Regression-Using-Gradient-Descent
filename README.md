# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Load necessary libraries for data handling, metrics, and visualization.

2.Load Data: Read the dataset using pd.read_csv() and display basic information.

3.Initialize Parameters: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4.Gradient Descent: Perform iterations to update m and c using gradient descent.

5.Plot Error: Visualize the error over iterations to monitor convergence of the model.
 
## Program and Output:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: DHARANYA.N
RegisterNumber: 212223230044
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
![Screenshot 2024-08-28 134409](https://github.com/user-attachments/assets/704028e3-e0da-4533-893d-254b5c0b3d2e)
```
dataset.info()
```
![Screenshot 2024-08-28 134521](https://github.com/user-attachments/assets/33a81b0b-ef77-4397-a30d-30da17009bb7)
```
#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)
```
![Screenshot 2024-08-28 134616](https://github.com/user-attachments/assets/0b814a08-cb70-4c11-8895-1894f5c1b56c)
```
m=0
c=0
L=0.001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
  Y_pred=m*X+c
  D_m=(-2/n)*sum(X*(Y-Y_pred))
  D_c=(-2/n)*sum(Y-Y_pred)
  m=m-L*D_m
  c=c-L*D_c
  error.append(sum(Y-Y_pred)**2)
print(m,c)
type(error)
print(len(error))
#print(error)
plt.plot(range(0,epochs),error)
```
![image](https://github.com/user-attachments/assets/95bc33f8-3b87-47e2-9523-65ae8d44094b)
![image](https://github.com/user-attachments/assets/5cff2066-be60-41e7-bf99-217ffc1ed0ac)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
