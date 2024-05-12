import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'Student_ID': [1, 2, 3, 4, 5],
    'Gender': ['M', 'F', 'M', 'F', 'M'],
    'Age': [18, 19, 20, np.nan, 22],
    'Math_Score': [85, 90, 75, 92, 88],
    'Physics_Score': [78, 82, 80, 75, 85],
    'English_Score': [82, 88, np.nan, 90, 84]
}

df = pd.DataFrame(data)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Replace missing values with mean or median
df['Age'].fillna(df['Age'].median(), inplace=True)
df['English_Score'].fillna(df['English_Score'].mean(), inplace=True)


# Define a function to detect outliers using IQR
def detect_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (data < lower_bound) | (data > upper_bound)

# Check for outliers in numeric variables
outliers = detect_outliers(df.select_dtypes(include=[np.number]))
print("Outliers:\n", outliers)

# Replace outliers with median
for col in outliers.columns:
    df[col] = np.where(outliers[col], df[col].median(), df[col])


# Apply log transformation to 'Age' variable
df['Age'] = np.log(df['Age'])
