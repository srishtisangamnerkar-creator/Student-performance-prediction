# Student Performance Prediction and At-Risk Analysis
# Standalone Python Script for IML Mini Project (2nd Year B.Tech)
# Uses train.csv (8000 records) and test.csv (2000 records)

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 70)
print("STUDENT PERFORMANCE PREDICTION AND AT-RISK ANALYSIS")
print("Course: Introduction to Machine Learning (IML) | 2nd Year B.Tech")
print("=" * 70)

# Step 1: Loading train and test datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print(f"\n1. Loaded Datasets:")
print(f"   - Training records: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"   - Testing records:  {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# Step 2: Data Cleaning and Preprocessing
for df in [train_df, test_df]:
    df['math_score'] = pd.to_numeric(
        df['math_score'].astype(str).str.replace(r'\\t|\t', '', regex=True), 
        errors='coerce'
    )
    median_math = train_df['math_score'].median()
    df['math_score'] = df['math_score'].fillna(median_math)

dept_mapping = {'CS': 0, 'Engineering': 1, 'Business': 2, 'Mathematics': 3}
for df in [train_df, test_df]:
    df['Gender_Code'] = (df['Gender'] == 'Female').astype(int)
    df['Department_Code'] = df['Department'].map(dept_mapping).fillna(0).astype(int)
    df['At_Risk'] = (df['Grade'] == 'F').astype(int)

feature_columns = [
    'Attendance (%)', 'Midterm_Score', 'Assignments_Avg', 'Quizzes_Avg',
    'Participation_Score', 'Projects_Score', 'test_preparation_course',
    'math_score', 'reading_score', 'writing_score', 'science_score',
    'Gender_Code', 'Department_Code'
]

X_train = train_df[feature_columns]
y_train = train_df['Final_Score']
X_test = test_df[feature_columns]
y_test = test_df['Final_Score']

print(f"\n2. Features Selected ({len(feature_columns)} features):")
print("   " + ", ".join(feature_columns))

# Step 3: Model 1 - Multiple Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Step 4: Model 2 - Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=20, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Step 5: Regression Evaluation
def get_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'Model': name, 'MAE': round(mae, 4), 'MSE': round(mse, 4), 'RMSE': round(rmse, 4), 'R2 Score': round(r2, 4)}

results = pd.DataFrame([
    get_metrics(y_test, y_pred_lr, 'Multiple Linear Regression'),
    get_metrics(y_test, y_pred_dt, 'Decision Tree Regressor (depth=5)')
])

print("\n3. Regression Evaluation on Test Set (test.csv - 2000 students):")
print(results.to_string(index=False))

# Step 6: Classification Task - At-Risk Student Detection
y_train_risk = train_df['At_Risk']
y_test_risk = test_df['At_Risk']

risk_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
risk_classifier.fit(X_train, y_train_risk)
y_pred_risk = risk_classifier.predict(X_test)

acc = accuracy_score(y_test_risk, y_pred_risk)
print(f"\n4. Classification Task (Decision Tree Classifier):")
print(f"   - Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_risk, y_pred_risk, target_names=['Safe (Grade A-D)', 'At-Risk (Grade F)']))

# Step 7: Weak Subject Diagnosis Function
print("5. Diagnostic Demonstration on Test Student Case:")
sample_student = test_df[test_df['Grade'] == 'F'].iloc[0]
pred_final = lr_model.predict(pd.DataFrame([sample_student])[feature_columns])[0]
weak_subs = [s for s in ['math_score', 'reading_score', 'writing_score', 'science_score'] if sample_student[s] < 50]
print(f"   Student ID: {sample_student['Student_ID']}")
print(f"   Department: {sample_student['Department']} | Attendance: {sample_student['Attendance (%)']:.1f}%")
print(f"   Predicted Final Exam Score: {pred_final:.2f} / 100")
print(f"   Actual Grade in Test Set: {sample_student['Grade']}")
print(f"   Weak Subjects Detected (<50): {', '.join(weak_subs)}")

# Step 8: Standalone Figures Verification
print("\n6. Figures Verified in figures/ directory:")
for f in sorted(os.listdir('figures')):
    fpath = os.path.join('figures', f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"   - figures/{f} ({size_kb:.1f} KB)")

print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 70)
