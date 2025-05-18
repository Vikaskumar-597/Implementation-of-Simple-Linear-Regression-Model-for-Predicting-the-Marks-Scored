# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries
2. Load the dataset
3. Extract independent (X) and dependent (y) variables
4. Split the dataset into training and testing sets
5. Create and train the linear regression model
6. Make predictions using the test data
7. Evaluate the model performance 
8. Visualize the regression line with training and test data


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by  : VIKASKUMAR M
RegisterNumber: 212224220122
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("/content/student_scores.csv")
df.head()
```
```
df.tail()
```
```
x = df.iloc[:,:-1].values
x
```
```
y = df.iloc[:,1].values
y
```
```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=1/3, random_state=0)
```
```
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```
```
y_pred = regressor.predict(X_test)
Y_test
```
```
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Training set")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Obtained")
plt.show()
```
```
plt.scatter(X_test, Y_test, color="purple")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Test set")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Obtained")
plt.show()
```
```
mse = mean_squared_error(Y_test, y_pred)
print("MSE =", mse)

mae = mean_absolute_error(Y_test, y_pred)
print("MAE =", mae)

rmse = np.sqrt(mse)
print("RMSE =", rmse)
```
## Output:

![image](https://github.com/user-attachments/assets/d84f5461-2895-4060-8ce7-41e553e50aab)

![image](https://github.com/user-attachments/assets/e2e630b6-1543-41c8-a211-d4ceaebdd9ef)

![image](https://github.com/user-attachments/assets/d85c3c74-0cfe-46b9-b284-eb1c16a1156c)

![image](https://github.com/user-attachments/assets/79952da7-6c1a-4a9b-b22b-06c1e84ad123)

![image](https://github.com/user-attachments/assets/b2269f5d-563a-4160-8cc5-e7623ad7694a)

![image](https://github.com/user-attachments/assets/d1bdd7d5-1391-442e-8e71-826a76e23cbe)

![image](https://github.com/user-attachments/assets/deeb2659-754d-4d8d-a4d7-217d20458d51)

![image](https://github.com/user-attachments/assets/2e243dc8-5692-4fc8-8c75-481259c31104)

![image](https://github.com/user-attachments/assets/5210ef65-69d3-4456-8a4d-2f4b8e0ff5f2)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
