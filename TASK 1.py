#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/pavanijain/Desktop/2013_nba_draft_combine.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display basic information about the dataset
print("\nBasic Information:")
print(data.info())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Checking for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualizations
# Correlation heatmap (only for numeric columns)
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(14, 10))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Distribution of numeric features
numeric_data.hist(bins=15, figsize=(20, 15), layout=(5, 4))
plt.tight_layout()
plt.show()

# Pairplot of selected features
selected_features = ['Height (No Shoes)', 'Height (With Shoes)', 'Wingspan', 'Standing reach', 'Vertical (Max)', 'Weight']
sns.pairplot(data[selected_features].dropna())
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.tight_layout()
plt.show()

# Boxplots of selected features
plt.figure(figsize=(20, 10))
for i, feature in enumerate(selected_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=data[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# Draft pick distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Draft pick', data=data, order=data['Draft pick'].dropna().value_counts().index)
plt.title('Draft Pick Distribution')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Vertical (Max) vs. Weight scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Weight', y='Vertical (Max)', data=data)
plt.title('Vertical (Max) vs. Weight')
plt.tight_layout()
plt.show()

# Standing reach vs. Wingspan scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Wingspan', y='Standing reach', data=data)
plt.title('Standing reach vs. Wingspan')
plt.tight_layout()
plt.show()


# In[ ]:




