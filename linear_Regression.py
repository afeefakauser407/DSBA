# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
boston_dataset = load_boston()

# Create a pandas DataFrame
boston_df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

# Add the target column 'MEDV' (median value of owner-occupied homes) to the DataFrame
boston_df['MEDV'] = boston_dataset.target

# Explore the dataset
print(boston_df.head())
print(boston_df.describe())

# Prepare the data for training
X = boston_df.drop('MEDV', axis=1)  # Features
y = boston_df['MEDV']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Training RMSE:", train_rmse)

y_test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("Testing RMSE:", test_rmse)

r2 = r2_score(y_test, y_test_pred)
print("R-squared score:", r2)

# Make predictions
new_data = pd.DataFrame({'CRIM': [0.1], 'ZN': [20.0], 'INDUS': [6.0], 'CHAS': [0.0], 'NOX': [0.5], 
                         'RM': [6.0], 'AGE': [50.0], 'DIS': [4.0], 'RAD': [4.0], 'TAX': [300.0], 
                         'PTRATIO': [15.0], 'B': [350.0], 'LSTAT': [10.0]})
predicted_price = model.predict(new_data)
print("Predicted price:", predicted_price)
