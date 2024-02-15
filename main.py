#Importing...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
var1 = pd.read_csv('heart_disease_uci.csv')

# Explore the data  
print(var1.info())
print(var1.shape)
print(var1.isnull().sum())

# Visualize the distribution of age
plt.figure(figsize=(10, 6))
sns.histplot(data=var1, x='age', hue='sex', element='step', stat='density')
plt.title('Distribution of Age')
plt.show()

# Calculate mean, median, and mode of age grouped by sex
mean_age_male = var1[var1['sex'] == 1]['age'].mean()
median_age_male = var1[var1['sex'] == 1]['age'].median()
mode_age_male = var1[var1['sex'] == 1]['age'].mode()[0]

mean_age_female = var1[var1['sex'] == 0]['age'].mean()
median_age_female = var1[var1['sex'] == 0]['age'].median()
mode_age_female = var1[var1['sex'] == 0]['age'].mode()[0]

print('Mean Age (Male):', mean_age_male)
print('Median Age (Male):', median_age_male)
print('Mode Age (Male):', mode_age_male)

print('Mean Age (Female):', mean_age_female)
print('Median Age (Female):', median_age_female)
print('Mode Age (Female):', mode_age_female)

# Impute missing values
imputer = IterativeImputer(random_state=42)
var1_imputed = pd.DataFrame(imputer.fit_transform(var1), columns=var1.columns)

# Detect and remove outliers
for col in var1.columns:
    if var1[col].dtype == 'float64':
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=var1_imputed, x='sex', y=col)
        plt.title(f'Box Plot of {col}')
        plt.show()

# Prepare the data for machine learning
X = var1_imputed.drop('target', axis=1)
y = var1_imputed['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the performance of several machine learning models
models = [
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Ada Boost', AdaBoostClassifier(random_state=42)),
    ('XG Boost', XGBClassifier(random_state=4)),]
