import pandas as pd

# Load the Titanic dataset
titanic_data = pd.read_csv("titanic.csv")

# Group by a categorical variable (e.g., 'Pclass') and calculate summary statistics for numeric variables (e.g., 'Age')
summary_stats = titanic_data.groupby('Pclass').agg({'Age': ['mean', 'median', 'min', 'max', 'std']})
print(summary_stats)


#Display basic statistical details for each species of iris dataset:
# Load the iris dataset
iris_data = pd.read_csv("iris.csv")  # Replace "iris.csv" with your dataset

# Filter data for each species
setosa_stats = iris_data[iris_data['Species'] == 'Iris-setosa'].describe()
versicolor_stats = iris_data[iris_data['Species'] == 'Iris-versicolor'].describe()
virginica_stats = iris_data[iris_data['Species'] == 'Iris-virginica'].describe()

print("Summary statistics for Iris-setosa:")
print(setosa_stats)

print("\nSummary statistics for Iris-versicolor:")
print(versicolor_stats)

print("\nSummary statistics for Iris-virginica:")
print(virginica_stats)

                                 
