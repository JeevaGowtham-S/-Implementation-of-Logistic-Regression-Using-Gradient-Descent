# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:

To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.
6. Obtain the graph.

## Program:

```py

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: JEEVAGOWTHAM S
RegisterNumber:  212222230053

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset= pd.read_csv("/content/Placement_Data (1).csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)



# catgorising col for further labelling
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset ["hsc_b"].astype('category')
dataset ["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset ["workex"].astype('category')
dataset ["specialisation"] = dataset ["specialisation"].astype('category')
dataset ["status"] = dataset ["status"].astype('category')
dataset["hsc_s"] = dataset ["hsc_s"].astype('category')
dataset.dtypes


# labelling the columns
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset ["hsc_b"].cat.codes
dataset["degree_t"] = dataset ["degree_t"].cat.codes
dataset["workex"] = dataset ["workex"].cat.codes
dataset["specialisation"] = dataset ["specialisation"].cat.codes
dataset["status"] = dataset ["status"].cat.codes
dataset["hsc_s"] = dataset ["hsc_s"].cat.codes
# display dataset
dataset

X =dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y

# Initialize the model parameters.
theta =np.random.randn(X.shape[1])
y=Y
# Define the sigmoid function.
def sigmoid(z):
      return 1 / (1 + np.exp(-z))

# Define the loss function.
def loss (theta, X, y):
      h = sigmoid(X.dot (theta))
      return -np.sum(y * np.log(h) + (1- y) * np.log(1-h))


def gradient_descent (theta, X, y, alpha, num_iterations):
      m = len(y)
      for i in range(num_iterations):
          h=sigmoid(X.dot(theta))
          gradient=X.T.dot(h-y) / m
          theta -= alpha*gradient
      return theta

theta =gradient_descent (theta, X, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
      h=sigmoid(X.dot(theta))
      y_pred =np.where (h >= 0.5, 1, 0)
      return y_pred

y_pred =predict (theta,X)

accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)

print(y_pred)

print(Y)


xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)

xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```

## Output:
## Dataset:
![Screenshot 2024-04-29 093307](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/88ac4023-910e-4121-93d9-6e384e333654)

## Dataset.dtypes:
![Screenshot 2024-04-29 093755](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/a35191ff-864c-47d3-8081-bda70211f2c9)

## Labeled_dataset:
![Screenshot 2024-04-29 093801](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/f354e51d-f40c-4550-8a0c-3b1ecbe51e9e)

## Dependent variable Y:
![Screenshot 2024-04-29 093808](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/9bf05190-a99a-405d-803a-0c80b40c3f3e)

## Accuracy:
![Screenshot 2024-04-29 093816](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/2d7d028b-a581-4e19-8c80-331bb0203f11)

## y_pred:
![Screenshot 2024-04-29 093820](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/10d8ec1d-e237-405f-83ef-b30db7dd8c9b)

## Y:
![Screenshot 2024-04-29 093824](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/70d84e44-adae-4694-bec4-29567432f6f6)

## y_pred:
![Screenshot 2024-05-01 072831](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/cc8911e4-106e-468e-8ba1-131b99769ff9)

![Screenshot 2024-05-01 072856](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/4f4c2d42-95ac-40ed-834c-9a1138d0a07a)

![Screenshot 2024-04-29 093828](https://github.com/JeevaGowtham-S/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118042624/0de83959-826e-4230-b738-9e3703332797)





## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
