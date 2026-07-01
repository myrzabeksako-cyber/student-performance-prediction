# Student Habits vs Academic Performance
## Final Project Report — Introduction to Machine Learning

---

## 1. Topic Quality & Novelty

### 1.1 Topic

**How do daily lifestyle habits affect a student's academic performance?**

This project investigates the relationship between a wide range of student habits — including study time, sleep, social media usage, diet quality, mental health, and more — and their final exam scores. We frame this as a **supervised regression problem**: given a set of behavioural and demographic features about a student, can we accurately predict their exam score?

### 1.2 Novelty & Relevance

Academic performance is typically evaluated after the fact, leaving students and educators without actionable, early-stage predictive signals. Most prior work focuses narrowly on study hours or attendance. **This project is distinctive in three ways:**

1. It models **15 heterogeneous features** simultaneously — spanning habits, lifestyle, environment, and demographics.
2. It provides a **comparative benchmark** of five regression algorithms on the same dataset, making it a reproducible ML reference for educational analytics.
3. The results are surfaced through an **interactive Streamlit application** allowing real-time personalised prediction — closing the gap between research and practical use.

### 1.3 Motivation

Universities are increasingly interested in early-warning systems to identify at-risk students. A lightweight ML model that uses observable behavioural signals (study hours, sleep, mental health) rather than invasive data collection could provide timely intervention opportunities. This project directly addresses that need.

---

## 2. Data Collection & Preparation

### 2.1 Dataset

| Property | Value |
|---|---|
| Source | Synthetic dataset generated to reflect realistic student behaviour distributions |
| Rows | 1 000 students |
| Columns | 16 (15 features + 1 target) |
| Target | `exam_score` (continuous, range 18.4–100.0) |

**Feature overview:**

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Student age (17–24) |
| `gender` | Categorical | Female / Male / Other |
| `study_hours_per_day` | Numeric | Daily hours of deliberate study |
| `social_media_hours` | Numeric | Daily hours on social media |
| `netflix_hours` | Numeric | Daily hours of video streaming |
| `part_time_job` | Binary | Has a part-time job (Yes/No) |
| `attendance_percentage` | Numeric | % classes attended |
| `sleep_hours` | Numeric | Hours of sleep per night |
| `diet_quality` | Ordinal | Poor / Fair / Good |
| `exercise_frequency` | Numeric | Days per week exercising |
| `parental_education_level` | Categorical | High School / Bachelor / Master |
| `internet_quality` | Ordinal | Poor / Average / Good |
| `mental_health_rating` | Numeric | Self-reported score 1–10 |
| `extracurricular_participation` | Binary | Participates in activities (Yes/No) |
| `exam_score` | Numeric | **Target variable** |

### 2.2 Data Cleaning

One feature contained missing values:

- `parental_education_level`: **91 missing values (9.1%)** — filled with the category `'Unknown'`, preserving all 1 000 rows.

No other missing values were found. No duplicate rows were detected.

### 2.3 Feature Construction

- **Categorical encoding:** All string features were transformed using `LabelEncoder` for tree-based models and `LabelEncoder` + `StandardScaler` for linear models.
- **Scaling:** Numeric features were standardised (zero mean, unit variance) before training Linear Regression and Ridge Regression, to ensure coefficient comparability and proper regularisation.
- **No feature was dropped:** All 14 non-ID, non-target columns were used. `student_id` was excluded as it carries no predictive signal.

---

## 3. Research & Data Analysis (EDA)

### 3.1 Key Insights

**Insight 1 — Study hours dominate.**
The correlation between `study_hours_per_day` and `exam_score` is **r = 0.83**, by far the strongest of any feature. This is reinforced by the Random Forest feature importance analysis, where `study_hours_per_day` accounts for the largest share of predictive variance.

**Insight 2 — Mental health has a meaningful positive effect.**
`mental_health_rating` correlates at **r = 0.32** with exam score — moderate but consistent. Students who report low mental health scores (≤3) score on average ~12 points lower than those who report high scores (≥8).

**Insight 3 — Screen time hurts, but modestly.**
Both `social_media_hours` and `netflix_hours` show negative correlations (both around **r = −0.17**). While the effect is not large, it points toward a time-displacement mechanism: screen time replaces study time.

**Insight 4 — Diet quality shows a clear ordinal effect.**
Median exam score rises from **~55** (Poor diet) to **~70** (Fair diet) to **~78** (Good diet), consistent with nutritional science literature linking diet to cognitive function.

**Insight 5 — Attendance matters less than expected.**
`attendance_percentage` correlates at only **r = 0.09** with exam score, suggesting that students who attend class but don't study independently still underperform. Active self-study is more important than passive attendance.

### 3.2 Visualisations

Four visualisations were produced (see `eda_overview.png`):

1. **Exam score histogram** — approximately normal distribution centred around 69.6, with a slight right skew.
2. **Study hours vs exam score scatter** — clear linear trend, coloured by mental health rating, revealing a compounding benefit of high study hours *and* good mental health.
3. **Correlation heatmap** — triangular heatmap of all numeric features; confirms dominance of `study_hours_per_day`.
4. **Box plots by diet quality** — median, IQR, and outlier spread across three diet groups.

### 3.3 Interpretation

The EDA confirms a **multi-causal model** of academic performance. No single habit explains everything; rather, performance is the product of a cluster of positive behaviours (study, sleep, mental health, diet) acting together. This justifies a multi-feature ML approach over simpler single-variable analysis.

---

## 4. Machine Learning Experiments

### 4.1 Algorithms Evaluated

| Model | Rationale |
|---|---|
| **Linear Regression** | Baseline; assumes linear additive relationships |
| **Ridge Regression** | Linear with L2 regularisation; controls for multicollinearity |
| **Decision Tree** | Non-linear, interpretable splits (max depth = 6) |
| **Random Forest** | Ensemble of trees; reduces variance via bagging |
| **Gradient Boosting** | Sequential ensemble; reduces bias via boosting |

### 4.2 Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **R²** | 1 − SS_res / SS_tot | Proportion of variance explained (higher is better, max 1.0) |
| **RMSE** | √(mean squared errors) | Average prediction error in score units (lower is better) |
| **MAE** | mean(|actual − predicted|) | Robust average absolute error (lower is better) |

**Train/test split:** 80% / 20% (800 training, 200 test), `random_state=42` for reproducibility.

### 4.3 Results

| Rank | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| 🥇 1 | **Linear Regression** | **0.8970** | **5.14** | **3.92** |
| 🥈 2 | Ridge Regression | 0.8969 | 5.14 | 3.92 |
| 🥉 3 | Gradient Boosting | 0.8774 | 5.61 | 4.37 |
| 4 | Random Forest | 0.8505 | 6.19 | 4.62 |
| 5 | Decision Tree | 0.6831 | 9.01 | 6.80 |

### 4.4 Comparison & Discussion

**Linear Regression achieves the best performance (R² = 0.897)**, meaning it explains ~90% of the variance in exam scores. This surprising result — where a simple linear model outperforms complex ensembles — is consistent with the EDA finding that `study_hours_per_day` has an overwhelmingly strong *linear* relationship with exam score.

- **Ridge Regression** performs nearly identically, confirming that multicollinearity is not a major issue in this dataset.
- **Gradient Boosting** (R² = 0.877) is the best non-linear model, capturing some interaction effects missed by linear models.
- **Random Forest** (R² = 0.851) shows slightly more variance due to its bootstrapping approach.
- **Decision Tree** (R² = 0.683) overfits individual splits despite depth restriction, producing the largest errors.

**Conclusion:** When the dominant signal is strongly linear, classical regression dominates. For practical deployment, **Linear Regression** is recommended: it is fast, interpretable, and generalisable.

---

## 5. Interactive ML Demonstration — Streamlit App

### 5.1 Functions

The app (`app.py`) implements four interactive pages:

| Page | Functionality |
|---|---|
| **Overview & EDA** | Dataset KPIs, score histogram, correlation heatmap, group box plots |
| **Feature Analysis** | Scatter plot with customisable colour axis, RF feature importance bar chart |
| **ML Experiments** | Model comparison table, actual-vs-predicted plots, residual distributions |
| **Score Predictor** | Real-time prediction form for all 14 input features |

### 5.2 Model Integration

- All five trained models are cached in memory using `@st.cache_data`.
- The predictor applies the appropriate preprocessing pipeline (scaling for linear models, raw values for tree models) before generating predictions.
- Predictions are clipped to [0, 100] to avoid out-of-range outputs.

### 5.3 User Interface

- **Dark theme** with a blue-cyan-purple colour palette for scientific clarity.
- **Sidebar navigation** between all four pages.
- **Personalised feedback**: after prediction, the app displays actionable tips if the user's habits fall below recommended thresholds (e.g., study < 3h, sleep < 6h).
- **Interactive controls**: sliders, dropdowns, and a model selector allow users to explore "what-if" scenarios.

### 5.4 How to Run

```bash
pip install streamlit scikit-learn pandas matplotlib seaborn
streamlit run app.py
```

Place `student_habits_performance.csv` in the same directory as `app.py`.

---

## Summary

This project demonstrates that student exam performance can be predicted with ~90% accuracy (R² = 0.897) using a simple Linear Regression model trained on 14 behavioural and demographic features. The dominant predictor is `study_hours_per_day`, followed by `mental_health_rating`. The findings suggest that academic interventions should prioritise increasing deliberate study time and supporting student mental wellbeing, rather than focusing on attendance or demographic factors alone.
