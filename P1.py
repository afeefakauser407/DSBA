import pandas as pd
import numpy as np


# Assuming you have downloaded the dataset and it is in the same directory as your script
data = pd.read_csv("iris.data", header=None) # Assuming the data file is named "iris.data"

# Check for missing values

missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Describe the dataset

description = data.describe()
print("\nDescription:\n", description)

# Variable descriptions
# The dataset consists of 5 columns: sepal length, sepal width, petal length, petal width, and class
# Variable descriptions:
# - Sepal length: Length of sepal in cm (numeric)
# - Sepal width: Width of sepal in cm (numeric)
# - Petal length: Length of petal in cm (numeric)
# - Petal width: Width of petal in cm (numeric)
# - Class: Species of iris flower (categorical)
# Dimensions of the dataframe
dimensions = data.shape
print("\nDimensions of DataFrame:", dimensions)


#Data Formatting and Data Normalization:
# Check data types of variables
data_types = data.dtypes
print("\nData Types:\n", data_types)

# If variables are not in the correct data type, apply proper type conversions.
# Since all variables are numeric, we don't need to apply type conversions in this case.

Turn categorical variables into quantitative variables in Python:
# Convert categorical variable 'Class' into quantitative variables using one-hot encoding
data = pd.get_dummies(data, columns=[4])

# Renaming the columns to something more meaningful
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_setosa', 'class_versicolor', 'class_virginica']
