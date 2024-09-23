# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. <b>Load and preprocess data:</b> Fetch dataset, create DataFrame, split into features and targets.
2. <b>Split and scale data:</b> Train-test split, apply standard scaling to features and targets.
3. <b>Initialize and train model:</b> Set up SGDRegressor with MultiOutputRegressor, fit on training data.
4. <b>Predict and evaluate:</b> Predict on test data, inverse transform, calculate mean squared error.

## Program And Outputs:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SETHUKKARASI C
RegisterNumber:  212223230201
*/
```

```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
```

```
# Load the California Housing dataset
data = fetch_california_housing()
print(data)
```

![output1](/o1.png)

```
import pandas as pd
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```

![output2](/o2.png)

```
df.info()
```

![output3](/o3.png)

```
X = df.drop(columns=['AveOccup','target'])
```

```
X.info()
```

![output4](/o4.png)

```
Y = df[['AveOccup' , 'target']]
```

```
Y.info()
```

![output5](/o5.png)

```
#Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=11)
X.head()
```

![output6](/o6.png)

```
#Scale the features and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
```

```
#Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

#Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

#Train the model
multi_output_sgd.fit(X_train, Y_train)
```

![output7](/o7.png)

```
# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
```

```
# Evaluate the model using Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Optionally, print some predictions
print("\nPredictions:\n", Y_pred[:5])  # Print first 5 predictions
```

![output8](/o8.png)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
