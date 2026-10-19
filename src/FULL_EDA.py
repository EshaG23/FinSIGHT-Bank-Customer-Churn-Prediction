# FinSIGHT - Intelligent Banking Customer Churn Prediction
# FULL_EDA.py: Tasks 2 to 5 (EDA, cleaning, preprocessing and splitting)
# Converted from the verified Jupyter notebook.

# Use the project folder for reading data and saving output files.
from pathlib import Path
PROJECT_FOLDER = Path(__file__).resolve().parent.parent


# =================================================================
# TASK 2 - EXPLORATORY DATA ANALYSIS (EDA)
# =================================================================

# Import the libraries used in our ML practicals.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme()

# Read the same CSV from the data folder in the VS Code project.
file_path = PROJECT_FOLDER / 'data' / 'Churn_Modelling.csv'
data = pd.read_csv(file_path)
print(data.head())

# Check dataset size, columns and types, just like in our practicals.
print('Shape:', data.shape)
print('Columns:', data.columns.tolist())
print('Data types:')
print(data.dtypes)
print('Dataset information:')
data.info()
print(data.tail())

# View numerical statistics and a summary of all columns.
print(data.describe())
print(data.describe(include='all'))

# Check how many customers stayed (0) and left (1).
print('Churn counts:')
print(data['Exited'].value_counts())
print('Churn percentages:')
print((data['Exited'].value_counts(normalize=True) * 100).round(2))

# Plot the target classes to understand class imbalance.
plt.figure(figsize=(6, 4))
sns.countplot(color='steelblue', x='Exited', data=data)
plt.title('Customers Who Stayed vs Left')
plt.xlabel('Exited (0 = Stayed, 1 = Left)')
plt.ylabel('Number of Customers')
plt.show()

# Study the age and credit score distributions.
plt.figure(figsize=(7, 4))
sns.histplot(color='steelblue', x='Age', data=data, bins=25, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(color='steelblue', x='CreditScore', data=data, bins=25, kde=True)
plt.title('Credit Score Distribution')
plt.show()

# Study account balance and estimated salary distributions.
plt.figure(figsize=(7, 4))
sns.histplot(color='steelblue', x='Balance', data=data, bins=25)
plt.title('Balance Distribution')
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(color='steelblue', x='EstimatedSalary', data=data, bins=25)
plt.title('Estimated Salary Distribution')
plt.show()

# Show how customers are distributed across countries and genders.
print(data['Geography'].value_counts())
print(data['Gender'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(color='steelblue', x='Geography', data=data)
plt.title('Customers by Geography')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(color='steelblue', x='Gender', data=data)
plt.title('Customers by Gender')
plt.show()

# Compare the percentage of customers leaving in each country.
print((data.groupby('Geography')['Exited'].mean() * 100).round(2))
plt.figure(figsize=(7, 4))
sns.barplot(color='steelblue', x='Geography', y='Exited', data=data)
plt.title('Average Churn Rate by Geography')
plt.ylabel('Churn Rate (0 to 1)')
plt.show()

# Compare churn across gender, products and active-member status.
for column in ['Gender', 'NumOfProducts', 'IsActiveMember']:
    print('\n', column)
    print((data.groupby(column)['Exited'].mean() * 100).round(2))
    plt.figure(figsize=(7, 4))
    sns.barplot(color='steelblue', x=column, y='Exited', data=data)
    plt.title('Average Churn Rate by ' + column)
    plt.ylabel('Churn Rate (0 to 1)')
    plt.show()

# Compare the ages of customers who stayed and left.
plt.figure(figsize=(7, 4))
sns.boxplot(color='steelblue', x='Exited', y='Age', data=data)
plt.title('Age vs Customer Churn')
plt.show()

# Compare average numerical values of customers who stayed and left.
# This helps us notice simple differences between the two groups.
compare_columns = ['Age', 'Balance', 'CreditScore', 'Tenure', 'EstimatedSalary', 'NumOfProducts']
print(data.groupby('Exited')[compare_columns].mean().round(2))

# A heatmap shows relationships among numerical columns.
# Remove ID columns here because they are not meaningful predictors.
numeric_data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
plt.figure(figsize=(11, 7))
sns.heatmap(numeric_data.select_dtypes(include=np.number).corr(), annot=True, cmap='bwr', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# =================================================================
# TASK 3 - DATA CLEANING
# =================================================================

# Look for missing values, duplicate rows and duplicate customer IDs.
print('Missing values:')
print(data.isnull().sum())
print('Duplicate rows:', data.duplicated().sum())
print('Duplicate CustomerIds:', data['CustomerId'].duplicated().sum())

# Remove exact duplicates if found, without changing the original data.
clean_data = data.copy()
clean_data.drop_duplicates(inplace=True)
print('Rows after cleaning:', clean_data.shape[0])

# Check valid ranges and target values before modelling.
print('Exited values:', clean_data['Exited'].unique())
print('Age range:', clean_data['Age'].min(), 'to', clean_data['Age'].max())
print('Balance below zero:', (clean_data['Balance'] < 0).sum())
print('Salary below zero:', (clean_data['EstimatedSalary'] < 0).sum())
print('Credit score below zero:', (clean_data['CreditScore'] < 0).sum())

# Remove identifiers and names because they should not predict churn.
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
model_data = clean_data.drop(columns=columns_to_drop)
print('Columns used for modelling:', model_data.columns.tolist())
print(model_data.head())


# =================================================================
# TASKS 4 AND 5 - PREPROCESSING AND TRAIN-TEST SPLIT
# =================================================================

# Separate customer attributes (X) from churn output (y).
X = model_data.drop(columns=['Exited'])
y = model_data['Exited']
print('Input shape:', X.shape)
print('Target shape:', y.shape)

# Split customers into 80% training and 20% testing.
# Stratify keeps roughly the same churn percentage in both groups.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print('Training input:', X_train.shape)
print('Testing input:', X_test.shape)
print('Training target:', y_train.shape)
print('Testing target:', y_test.shape)

# One-hot encode Geography and Gender using the training categories.
# Match test columns to the training columns so both have the same inputs.
X_train_encoded = pd.get_dummies(X_train, columns=['Geography', 'Gender']).astype(float)
X_test_encoded = pd.get_dummies(X_test, columns=['Geography', 'Gender']).astype(float)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
print('Encoded columns:', X_train_encoded.columns.tolist())
print('Encoded training shape:', X_train_encoded.shape)
print('Encoded testing shape:', X_test_encoded.shape)
print(X_train_encoded.head())

# Scale numeric features for distance-based models like KNN.
# Learn the mean and standard deviation from training data only.
from sklearn.preprocessing import StandardScaler

scale_columns = ['CreditScore', 'Age', 'Tenure', 'Balance',
                 'NumOfProducts', 'EstimatedSalary']
scaler = StandardScaler()
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()
X_train_scaled[scale_columns] = scaler.fit_transform(X_train_encoded[scale_columns])
X_test_scaled[scale_columns] = scaler.transform(X_test_encoded[scale_columns])
print(X_train_scaled.head())

# Check that train and test have the same columns and no missing values.
print('Train and test columns identical:', X_train_scaled.columns.equals(X_test_scaled.columns))
print('Missing values in encoded training data:', X_train_encoded.isnull().sum().sum())
print('Missing values in encoded testing data:', X_test_encoded.isnull().sum().sum())

# Compare class percentages so both sets represent the churn distribution.
class_percentages = pd.DataFrame({
    'Training (%)': y_train.value_counts(normalize=True).sort_index() * 100,
    'Testing (%)': y_test.value_counts(normalize=True).sort_index() * 100
})
print('Churn distribution in training and testing data:')
print(class_percentages.round(2))


# =================================================================
# EXPORT OUTPUTS FOR THE TEAM
# =================================================================

# Save the same train-test datasets for all three members.
# Also save the fitted scaler and feature names for the future app.
import joblib

output_folder = PROJECT_FOLDER / 'outputs'
output_folder.mkdir(parents=True, exist_ok=True)

X_train_encoded.to_csv(output_folder / 'X_train_encoded.csv', index=False)
X_test_encoded.to_csv(output_folder / 'X_test_encoded.csv', index=False)
X_train_scaled.to_csv(output_folder / 'X_train_scaled.csv', index=False)
X_test_scaled.to_csv(output_folder / 'X_test_scaled.csv', index=False)
y_train.to_csv(output_folder / 'y_train.csv', index=False)
y_test.to_csv(output_folder / 'y_test.csv', index=False)

joblib.dump(scaler, output_folder / 'scaler.pkl')
joblib.dump(X_train_encoded.columns.tolist(), output_folder / 'feature_columns.pkl')
joblib.dump(scale_columns, output_folder / 'scale_columns.pkl')

print('Saved 6 CSV files and 3 preprocessing files inside outputs.')

