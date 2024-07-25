import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv('Salary_data1.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=0)

#training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test result
y_pred=regressor.predict(X_test)

#visualising test set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#visulaising test set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue') #regression line will be same for training set values and test set values
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))

#Getting the linear regression equaltion with coefficient values
print(regressor.coef_)
print(regressor.intercept_)
#Therefore, the equation of our simple linear regression model is:

#Salary=9345.94×YearsExperience+26816.19 

#Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values.