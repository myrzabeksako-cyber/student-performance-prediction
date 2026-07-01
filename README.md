# 🎓 Student Academic Performance Prediction

A complete Machine Learning project that predicts student exam scores based on lifestyle, behavioral, and demographic factors.

The project covers the entire data science workflow:

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Preprocessing
- 🤖 Machine Learning Model Comparison
- 📈 Feature Importance Analysis
- 🌐 Interactive Streamlit Web Application

---

# Project Overview

The objective of this project is to investigate how different student habits affect academic performance and to build machine learning models capable of predicting exam scores.

The dataset includes information about study habits, sleep, diet quality, mental health, attendance, social media usage, and other lifestyle factors.

Five regression algorithms were trained and evaluated using standard regression metrics.

---

# Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

# Project Structure

```
student-performance-prediction/

│
├── app.py
├── README.md
├── report.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── student_habits_performance.csv
│
└── figures/
    ├── eda_overview.png
    └── ml_results.png
```

---

# Dataset

The dataset contains **1000 student records** with multiple features describing academic and lifestyle habits.

Features include:

- Age
- Gender
- Study Hours
- Sleep Hours
- Social Media Usage
- Netflix Usage
- Attendance
- Diet Quality
- Mental Health Rating
- Exercise Frequency
- Internet Quality
- Part-time Job
- Extracurricular Activities
- Parental Education
- Exam Score (Target)

---

# Data Preprocessing

The following preprocessing steps were applied:

- Missing value handling
- Categorical encoding
- Feature scaling
- Feature engineering
- Train/Test Split
- Data validation

---

# Exploratory Data Analysis

The project includes multiple visualizations to better understand the dataset.

### EDA Dashboard

![EDA](figures/eda_overview.png)

The analysis includes:

- Exam score distribution
- Study hours vs exam score
- Correlation heatmap
- Diet quality comparison

---

# Machine Learning Models

The following regression algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

---

# Model Performance

| Model | R² Score | RMSE | MAE |
|-------|---------:|------:|------:|
| Linear Regression | **0.897** | **5.14** | **3.92** |
| Ridge Regression | 0.897 | 5.14 | 3.92 |
| Gradient Boosting | 0.877 | 5.61 | 4.37 |
| Random Forest | 0.851 | 6.19 | 4.62 |
| Decision Tree | 0.683 | 9.01 | 6.80 |

**Best Model:** Linear Regression

---

# Machine Learning Results

![ML Results](figures/ml_results.png)

The comparison includes:

- Model R² Scores
- RMSE & MAE Comparison
- Random Forest Feature Importance

---

# Feature Importance

The most influential variables for predicting exam scores were:

1. Study Hours per Day
2. Mental Health Rating
3. Social Media Usage
4. Sleep Hours
5. Netflix Usage

Study time was identified as the strongest predictor of academic performance.

---

# Streamlit Application

The project also includes an interactive Streamlit application where users can:

- Explore the dataset
- View EDA visualizations
- Compare machine learning models
- Predict exam scores based on custom inputs

Run the application:

```bash
streamlit run app.py
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/student-performance-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit app:

```bash
streamlit run app.py
```

---

# Future Improvements

Possible future enhancements include:

- Hyperparameter tuning
- Cross-validation
- XGBoost implementation
- LightGBM implementation
- Model deployment
- Docker support
- Cloud deployment

---

# Author

**Myrzabek Sagynzhan**

Data Science Student

---

# Disclaimer

This project was created for educational and portfolio purposes.
