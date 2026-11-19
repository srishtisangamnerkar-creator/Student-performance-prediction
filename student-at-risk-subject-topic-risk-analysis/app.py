# -*- coding: utf-8 -*-
"""
Student Performance and Risk Predictor
Interactive Web Application for Machine Learning Mini Project
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

st.set_page_config(
    page_title="Student Performance and Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Friendly, simple styling (No badges, no emoticons)
st.markdown("""
<style>
    .header-bar {
        background-color: #4f46e5;
        color: white;
        padding: 18px 22px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .header-bar h1 {
        color: white;
        font-size: 24px;
        margin: 0;
        font-weight: bold;
    }
    .header-bar p {
        color: #e0e7ff;
        font-size: 14px;
        margin: 6px 0 0 0;
    }
    .status-safe {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: bold;
        border: 1px solid #bbf7d0;
        display: inline-block;
    }
    .status-risk {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: bold;
        border: 1px solid #fecaca;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Train models
@st.cache_resource
def load_and_train_models():
    train_path = "train.csv"
    test_path = "test.csv"
    if not os.path.exists(train_path):
        train_path = os.path.join(os.path.dirname(__file__), "train.csv")
        test_path = os.path.join(os.path.dirname(__file__), "test.csv")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    for df in [train_df, test_df]:
        df['math_score'] = pd.to_numeric(
            df['math_score'].astype(str).str.replace(r'\\t|\t', '', regex=True),
            errors='coerce'
        ).fillna(58.0)
        df['Gender_Code'] = (df['Gender'] == 'Female').astype(int)
        dept_map = {'CS': 0, 'Engineering': 1, 'Business': 2, 'Mathematics': 3}
        df['Department_Code'] = df['Department'].map(dept_map).fillna(0).astype(int)
        df['At_Risk'] = (df['Grade'] == 'F').astype(int)

    feature_cols = [
        'Attendance (%)', 'Midterm_Score', 'Assignments_Avg', 'Quizzes_Avg',
        'Participation_Score', 'Projects_Score', 'test_preparation_course',
        'math_score', 'reading_score', 'writing_score', 'science_score',
        'Gender_Code', 'Department_Code'
    ]

    X_train = train_df[feature_cols]
    y_train = train_df['Final_Score']

    lr = LinearRegression().fit(X_train, y_train)
    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=20, random_state=42).fit(X_train, y_train)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, train_df['At_Risk'])

    return train_df, test_df, lr, dt, clf, feature_cols

train_df, test_df, lr_model, dt_model, clf_model, feature_cols = load_and_train_models()

# Clean Header without the 3 removed topics
st.markdown("""
<div class="header-bar">
    <h1>Student Performance and Risk Predictor</h1>
    <p>Adjust student marks and attendance on the left to predict final scores and determine risk status.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.markdown("### Student Academic Factors")
preset = st.sidebar.radio("Quick Presets:", ["Custom Input", "Top Student", "Average Student", "At-Risk Student"])

if preset == "Top Student":
    def_att, def_mid, def_asg, def_qz, def_part, def_proj = 95.0, 92.0, 90.0, 88.0, 85.0, 94.0
    def_math, def_read, def_write, def_sci = 90.0, 88.0, 86.0, 92.0
    def_dept_idx, def_prep_idx = 0, 1
elif preset == "Average Student":
    def_att, def_mid, def_asg, def_qz, def_part, def_proj = 75.0, 65.0, 70.0, 68.0, 60.0, 72.0
    def_math, def_read, def_write, def_sci = 62.0, 68.0, 65.0, 64.0
    def_dept_idx, def_prep_idx = 1, 0
elif preset == "At-Risk Student":
    def_att, def_mid, def_asg, def_qz, def_part, def_proj = 52.0, 38.0, 45.0, 40.0, 35.0, 50.0
    def_math, def_read, def_write, def_sci = 36.0, 42.0, 48.0, 44.0
    def_dept_idx, def_prep_idx = 2, 0
else:
    def_att, def_mid, def_asg, def_qz, def_part, def_proj = 82.0, 70.0, 75.0, 72.0, 68.0, 74.0
    def_math, def_read, def_write, def_sci = 65.0, 70.0, 68.0, 72.0
    def_dept_idx, def_prep_idx = 0, 1

attendance = st.sidebar.slider("Attendance Rate (%)", 30.0, 100.0, def_att, 1.0)
midterm = st.sidebar.slider("Midterm Exam Score", 10.0, 100.0, def_mid, 1.0)
assignments = st.sidebar.slider("Assignment Average", 20.0, 100.0, def_asg, 1.0)
quizzes = st.sidebar.slider("Quizzes Average", 20.0, 100.0, def_qz, 1.0)
participation = st.sidebar.slider("Participation Score", 10.0, 100.0, def_part, 1.0)
projects = st.sidebar.slider("Projects Score", 10.0, 100.0, def_proj, 1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Foundational Subject Scores:**")
math_s = st.sidebar.slider("Mathematics", 10.0, 100.0, def_math, 1.0)
reading_s = st.sidebar.slider("Reading", 10.0, 100.0, def_read, 1.0)
writing_s = st.sidebar.slider("Writing", 10.0, 100.0, def_write, 1.0)
science_s = st.sidebar.slider("Science", 10.0, 100.0, def_sci, 1.0)

st.sidebar.markdown("---")
dept = st.sidebar.selectbox("Department", ["Computer Science", "Engineering", "Business", "Mathematics"], index=def_dept_idx)
prep_course = st.sidebar.selectbox("Test Preparation Course", ["Completed", "None"], index=def_prep_idx)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

dept_code = {"Computer Science": 0, "Engineering": 1, "Business": 2, "Mathematics": 3}[dept]
gender_code = 1 if gender == "Female" else 0
prep_code = 1 if prep_course == "Completed" else 0

# Vector
input_dict = {
    'Attendance (%)': attendance,
    'Midterm_Score': midterm,
    'Assignments_Avg': assignments,
    'Quizzes_Avg': quizzes,
    'Participation_Score': participation,
    'Projects_Score': projects,
    'test_preparation_course': prep_code,
    'math_score': math_s,
    'reading_score': reading_s,
    'writing_score': writing_s,
    'science_score': science_s,
    'Gender_Code': gender_code,
    'Department_Code': dept_code
}

input_df = pd.DataFrame([input_dict])[feature_cols]

pred_lr = float(lr_model.predict(input_df)[0])
pred_dt = float(dt_model.predict(input_df)[0])
pred_risk = int(clf_model.predict(input_df)[0])

is_flagged_risk = (pred_risk == 1 or attendance < 60.0 or midterm < 45.0 or pred_lr < 50.0)

# Top 3 KPI Cards (Diagnostic assessment removed)
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric("Linear Regression Score", f"{pred_lr:.1f} / 100", help="Predicted final exam score using Multiple Linear Regression")

with kpi2:
    st.metric("Decision Tree Score", f"{pred_dt:.1f} / 100", help="Predicted final exam score using Decision Tree Regressor (Depth=5)")

with kpi3:
    st.markdown("**Risk Status**")
    if is_flagged_risk:
        st.markdown('<div class="status-risk">AT-RISK</div>', unsafe_allow_html=True)
        st.caption("Needs academic counseling and tutoring")
    else:
        st.markdown('<div class="status-safe">SAFE</div>', unsafe_allow_html=True)
        st.caption("Passing academic standing")

st.markdown("---")

# Main Tabs (Diagnostic and Viva Voce removed)
tab_pred, tab_eval, tab_feat, tab_vis = st.tabs([
    "Prediction Summary",
    "Model Evaluation & Comparison",
    "Feature Importance",
    "Project Visualizations"
])

with tab_pred:
    st.subheader("Student Prediction Summary")
    
    st.markdown(f"""
    - **Linear Regression Prediction:** **{pred_lr:.1f} / 100**
    - **Decision Tree Regressor Prediction:** **{pred_dt:.1f} / 100**
    - **Estimated Grade Bracket:** **{'Grade A' if pred_lr >= 80 else 'Grade B' if pred_lr >= 70 else 'Grade C' if pred_lr >= 60 else 'Grade D' if pred_lr >= 50 else 'Grade F'}**
    - **Attendance Status:** **{attendance:.1f}% ({'Satisfactory' if attendance >= 75 else 'Low Attendance Warning'})**
    """)
    
    st.markdown("#### Mathematical Formulation (OLS Equation)")
    st.code("""Final_Score = 68.1344 - (0.0107 * Attendance) + (0.0031 * Midterm) + (0.0128 * Assignments) 
            - (0.0006 * Quizzes) + (0.0187 * Participation) - (0.0002 * Projects) 
            + (0.0903 * PrepCourse) - (0.0002 * Math) - (0.0098 * Reading) 
            + (0.0115 * Writing) + (0.0134 * Science) + (0.2856 * Department)""", language="python")

with tab_eval:
    st.subheader("Model Evaluation Metrics on Unseen Test Dataset (2,000 Records)")
    
    comp_df = pd.DataFrame([
        {"Model": "Multiple Linear Regression", "MAE": 15.239, "MSE": 305.321, "RMSE": 17.473, "R2 Score": -0.0010},
        {"Model": "Decision Tree Regressor (depth=5)", "MAE": 15.291, "MSE": 309.448, "RMSE": 17.591, "R2 Score": -0.0145}
    ])
    st.table(comp_df)
    
    st.markdown("#### Error Metric Comparison (Lower is Better)")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    sns.barplot(x="Model", y="MAE", data=comp_df, ax=ax[0], palette=["#4f46e5", "#10b981"])
    ax[0].set_title("Mean Absolute Error (MAE)")
    ax[0].set_ylabel("Error in Marks")
    
    sns.barplot(x="Model", y="RMSE", data=comp_df, ax=ax[1], palette=["#4f46e5", "#10b981"])
    ax[1].set_title("Root Mean Squared Error (RMSE)")
    ax[1].set_ylabel("Error in Marks")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("""
    **Model Comparison Takeaways:**
    - Multiple Linear Regression achieved slightly lower error (MAE 15.239 vs. 15.291 marks, RMSE 17.473 vs. 17.591 marks).
    - Continuous examination scores are better approximated by smooth linear trends than step-wise tree partitions.
    - The Decision Tree Classifier attained 83.20% accuracy when flagging at-risk students.
    """)

with tab_feat:
    st.subheader("Feature Importance for Identifying At-Risk Students")
    st.write("Ranked by Gini Impurity reduction from Decision Tree Classifier (max_depth=4):")
    
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": clf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    fig_imp, ax_imp = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x="Importance", y="Feature", data=importances, palette="mako", ax=ax_imp)
    ax_imp.set_title("Gini Feature Importance")
    plt.tight_layout()
    st.pyplot(fig_imp)

with tab_vis:
    st.subheader("Generated Project Visualizations")
    fig_dir = "figures"
    if os.path.exists(fig_dir):
        fig_files = sorted([f for f in os.listdir(fig_dir) if f.endswith('.png')])
        for fig_f in fig_files:
            st.image(os.path.join(fig_dir, fig_f), caption=fig_f, use_container_width=True)
