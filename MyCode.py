Below is the complete Python code for the predictive modeling project, including all steps in a single file. You can save it as a `.py` file and upload it to GitHub.

```python
# customer_retention_model.py

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 1: Data Overview
print("Dataset Overview:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 2: Handle Missing Values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

# Step 3: Exploratory Data Analysis
# Churn distribution
sns.countplot(data['Churn'])
plt.title("Churn Distribution")
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
plt.title("Monthly Charges vs Churn")
plt.show()

# Step 4: Data Preprocessing
# Encode categorical variables
label_enc = LabelEncoder()
for column in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
               'PaperlessBilling', 'PaymentMethod', 'Churn']:
    data[column] = label_enc.fit_transform(data[column])

# Feature Scaling
scaler = StandardScaler()
data[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['MonthlyCharges', 'TotalCharges']])

# Splitting the data
X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Step 8: Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()

# Step 9: Save the Model
joblib.dump(model, "customer_retention_model.pkl")
print("Model saved as 'customer_retention_model.pkl'.")

# Instructions for using the model
print("\nHow to Use the Model:")
print("1. Load the model using joblib: `model = joblib.load('customer_retention_model.pkl')`")
print("2. Prepare input data in the same format as the training data.")
print("3. Use `model.predict()` to make predictions.")
```

### Instructions to Save and Upload to GitHub  

1. Save the code as `customer_retention_model.py`.  
2. Create a new GitHub repository.  
3. Add the file to your repository:  
   ```bash
   git init
   git add customer_retention_model.py
   git commit -m "Added customer retention predictive model"
   git branch -M main
   git remote add origin <your-repository-url>
   git push -u origin main
   ```

This script includes all steps of the project, from data loading to saving the model, and is ready to be used or shared. Let me know if you need help with GitHub commands or additional details!
