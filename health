import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv(url, names=columns)

# Replace missing values marked by '?' with NaN and drop them
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Convert columns to appropriate data types
df = df.astype(float)

# Display the first few rows of the dataset
print(df.head())

# Pair Plot of Selected Features
sns.pairplot(df, hue='target', vars=['age', 'trestbps', 'chol', 'thalach'])
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()

# Box Plot of Feature Distributions by Target
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
sns.boxplot(x='target', y='age', data=df)
plt.title('Age by Target')

plt.subplot(2, 2, 2)
sns.boxplot(x='target', y='trestbps', data=df)
plt.title('Resting Blood Pressure by Target')

plt.subplot(2, 2, 3)
sns.boxplot(x='target', y='chol', data=df)
plt.title('Cholesterol by Target')

plt.subplot(2, 2, 4)
sns.boxplot(x='target', y='thalach', data=df)
plt.title('Maximum Heart Rate by Target')

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()

# Data Preprocessing
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# T-test for Age Differences
no_disease = df[df['target'] == 0]['age']
disease = df[df['target'] > 0]['age']
t_stat_age, p_val_age = ttest_ind(no_disease, disease)
print(f"T-test for Age: t-statistic={t_stat_age}, p-value={p_val_age}")

# ANOVA Test for Differences in Cholesterol Levels
anova_chol = f_oneway(df[df['target'] == 0]['chol'], df[df['target'] == 1]['chol'],
                      df[df['target'] == 2]['chol'], df[df['target'] == 3]['chol'],
                      df[df['target'] == 4]['chol'])
print(f"ANOVA test for Cholesterol: F-statistic={anova_chol.statistic}, p-value={anova_chol.pvalue}")

# Chi-Square Test for Association between Gender and Heart Disease
contingency_table = pd.crosstab(df['sex'], df['target'])
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"Chi-Square Test: chi2={chi2}, p-value={p}")

# Additional Graphs
# Histogram of Age Distribution by Target
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=20, kde=True)
plt.title('Age Distribution by Target')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Violin Plot of Cholesterol by Chest Pain Type
plt.figure(figsize=(10, 6))
sns.violinplot(x='cp', y='chol', hue='target', data=df, split=True)
plt.title('Cholesterol by Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.ylabel('Cholesterol')
plt.show()

# Joint Plot of Age vs. Maximum Heart Rate
sns.jointplot(x='age', y='thalach', data=df, kind='scatter', hue='target')
plt.suptitle('Age vs. Maximum Heart Rate', y=1.02)
plt.show()

# Bar Plot of the Number of Major Vessels by Target
plt.figure(figsize=(10, 6))
sns.countplot(x='ca', hue='target', data=df)
plt.title('Number of Major Vessels by Target')
plt.xlabel('Number of Major Vessels')
plt.ylabel('Count')
plt.show()

# Line Plot of Cholesterol over Age
plt.figure(figsize=(10, 6))
sns.lineplot(x='age', y='chol', hue='target', data=df, ci=None)
plt.title('Cholesterol over Age')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.show()
