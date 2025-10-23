# Barra

Barra is a package for Linear Regression.

<div align="center">
    <img src="/pic/barra-fix.png" alt="Barra Image Formula"/>
</div>

```python

# Import library
from barra import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as  plt

# Read
df = pd.read_csv("student_scores.csv")
df.head()

plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.show()

# X dan y 
X = df[['Hours']]
y = df['Scores']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Predict
my_model = LinearRegression()
my_model.fit(X_train, y_train)
y_pred_my_model = my_model.predict(X_test)

print(f"My Model Prediction: {y_pred_my_model}")
```

```bash
My Model Prediction: [array([83.18814104]), array([27.03208774]), array([27.03208774]), array([69.63323162]), array([59.95115347])]
```

```python
r2 = r2_score(y_test, y_pred_my_model)
mae = mean_absolute_error(y_test, y_pred_my_model)
mse = mean_squared_error(y_test, y_pred_my_model)
rmse = np.sqrt(mse)

print("R2:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
```

```bash
R2: 0.9678055545167994
MAE: 3.9207511902099226
MSE: 18.943211722315247
RMSE: 4.352380006653285
```

### Sklearn Model

```python
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred_sklearn = model.predict(X_test)

print(f"Sklearn Prediciton: {y_pred_sklearn}")
```

```bash
Sklearn Prediciton: [83.18814104 27.03208774 27.03208774 69.63323162 59.95115347]
```

```python
r2 = r2_score(y_test, y_pred_sklearn)
mae = mean_absolute_error(y_test, y_pred_sklearn)
mse = mean_squared_error(y_test, y_pred_sklearn)
rmse = np.sqrt(mse)

print("R2:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
```

```bash
R2: 0.9678055545167994
MAE: 3.9207511902099244
MSE: 18.943211722315272
RMSE: 4.352380006653288
```

[Code](/barra/linreg.py) <br/>
[Barra Test and Compare](/barra.ipynb) <br/>
[See on pdf](/pic/Barra%20-%20Linear%20Regression%20From%20Scratch%20-%20Fix.pdf)

