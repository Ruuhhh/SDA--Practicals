# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data.csv")

# Display first 5 rows
print(data.head())

# Dataset information
print(data.info())

# Check missing values
print(data.isnull().sum())

# Handle missing values using mean
data.fillna(data.mean(numeric_only=True), inplace=True)

# Descriptive statistics
print(data.describe())

# Outlier detection using IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalization
data_norm = (data - data.mean()) / data.std()

# Histogram
data_norm.hist(figsize=(10,8))
plt.show()

# Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(data=data_norm)
plt.show()

# Scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=data_norm.iloc[:,0], y=data_norm.iloc[:,1])
plt.show()
