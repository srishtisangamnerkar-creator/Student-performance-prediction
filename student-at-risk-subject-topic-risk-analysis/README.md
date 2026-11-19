# Student Performance Prediction and At-Risk Analysis (IML Mini Project)

## Project Overview
This project is an Introduction to Machine Learning (IML) mini project for 2nd Year B.Tech students based on:
- **Topic 1:** Student Performance Prediction (Linear Regression / Decision Tree)
- **Dataset:** Kaggle Student At-Risk Subject and Topic Risk Analysis competition
- **Training Set:** `train.csv` (8,000 student records)
- **Testing Set:** `test.csv` (2,000 student records)

The project implements two core machine learning tasks:
1. **Regression Task:** Predicting continuous final examination scores (`Final_Score`) using Multiple Linear Regression and Decision Tree Regressor.
2. **Classification Task:** Classifying students at risk of failure (`Grade == 'F'`) using a Decision Tree Classifier and pinpointing specific weak foundational subjects (Math, Reading, Writing, Science < 50 marks) for early intervention.

---

## Three Ways to Experience the Single-Screen Dashboard

### 1. Standalone Single Web Page (`dashboard.html`)
The simplest and most impressive way to present the project:
- **How to Open:** Simply **double-click `dashboard.html`** in Windows Explorer (or open it with Chrome / Edge).
- **Features on a Single Screen:**
  - Real-time sliders for student academic parameters (Attendance, Midterm, Quizzes, Assignments, Projects, Math, Reading, Writing, Science).
  - Instant live dual score predictions: Multiple Linear Regression (OLS) vs. Decision Tree Regressor.
  - Color-coded Risk Status: `[SAFE / ON TRACK]` vs. `[AT-RISK / INTERVENTION REQUIRED]`.
  - Granular subject audit pinpointing scores below 50 marks.
  - Interactive Chart.js tabs comparing MAE, RMSE, R2, and Gini feature importances.
  - Viva Voce cheat sheet accordion.

### 2. Single-Screen Interactive Widget in Jupyter Notebook (`Student_Performance_Prediction.ipynb`)
Inside the notebook, Step 11 contains:
- An interactive `ipywidgets` control panel on a single screen where you can adjust sliders and view live updates.
- An embedded web view of `dashboard.html` directly in the notebook output tab.

### 3. Full Streamlit Web Application (`app.py`)
To run as a local Python web app:
```powershell
cd "C:\Users\Chanchal\Downloads\student_performance_prediction"
streamlit run app.py
```
This launches a browser interface at `http://localhost:8501`.

---

## Files in this Directory
- `dashboard.html`: Standalone single-screen interactive web dashboard (open with any browser).
- `app.py`: Streamlit Python web dashboard.
- `Student_Performance_Prediction.ipynb`: Jupyter Notebook with complete analysis and embedded single-screen widget tab.
- `train.csv`: Official training dataset (8,000 rows x 21 columns).
- `test.csv`: Official testing dataset (2,000 rows x 21 columns).
- `run_project.py`: Standalone Python script executing the full pipeline in terminal.
- `figures/`: Folder containing all 7 standalone 300 DPI visualization figures.
- `PROJECT_REPORT.md`: Formal academic project report ready for college submission.
- `VIVA_QUESTIONS_AND_ANSWERS.md`: 11 viva questions with student-friendly answers.
