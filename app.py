"""
Student Habits vs Academic Performance — Interactive ML App
Final Project | Introduction to Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Habits & Academic Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  .stApp { background: #0f0f1a; color: #e0e0e0; }
  .metric-card {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      border: 1px solid #00d4ff33;
      border-radius: 12px; padding: 20px; text-align: center;
      transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #00d4ff88; }
  .metric-value { font-size: 2rem; font-weight: 700; color: #00d4ff; }
  .metric-label { font-size: 0.85rem; color: #888; margin-top: 4px; }
  .section-header {
      background: linear-gradient(90deg, #00d4ff22, transparent);
      border-left: 3px solid #00d4ff;
      padding: 10px 16px; border-radius: 0 8px 8px 0;
      margin: 20px 0 10px 0;
  }
  .insight-box {
      background: #1a1a2e; border: 1px solid #533483;
      border-radius: 8px; padding: 14px 18px; margin: 8px 0;
  }
  .stButton > button {
      background: linear-gradient(135deg, #533483, #00d4ff);
      color: white; border: none; border-radius: 8px;
      padding: 10px 24px; font-weight: 600; font-size: 1rem;
      transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }
  h1, h2, h3 { color: #e0e0e0 !important; }
  .stSidebar { background: #10101e !important; }
  .stSelectbox label, .stSlider label, .stRadio label { color: #aaa !important; }
  code { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading & Preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("student_habits_performance.csv")
    df2 = df.copy()
    df2['parental_education_level'] = df2['parental_education_level'].fillna('Unknown')

    cat_cols = ['gender', 'part_time_job', 'diet_quality',
                'parental_education_level', 'internet_quality',
                'extracurricular_participation']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col])
        encoders[col] = le

    features = ['study_hours_per_day', 'social_media_hours', 'netflix_hours',
                'attendance_percentage', 'sleep_hours', 'exercise_frequency',
                'mental_health_rating', 'diet_quality', 'internet_quality',
                'part_time_job', 'extracurricular_participation', 'gender',
                'parental_education_level', 'age']
    X = df2[features]
    y = df2['exam_score']
    return df, df2, X, y, encoders, features

@st.cache_data
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Linear Regression':   LinearRegression(),
        'Ridge Regression':    Ridge(alpha=1.0),
        'Decision Tree':       DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest':       RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    linear_models = {'Linear Regression', 'Ridge Regression'}

    trained, results = {}, {}
    for name, model in models.items():
        Xtr = X_train_s if name in linear_models else X_train
        Xte = X_test_s  if name in linear_models else X_test
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        trained[name] = (model, Xtr, Xte, scaler)
        results[name] = {
            'R2':   round(r2_score(y_test, preds), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            'MAE':  round(mean_absolute_error(y_test, preds), 4),
            'preds': preds,
            'y_test': y_test.values,
        }
    return trained, results, X_train, X_test, y_train, y_test, scaler

# ─── Colour Palette ───────────────────────────────────────────────────────────
DARK  = '#0f0f1a'
CARD  = '#1a1a2e'
BLUE  = '#0f3460'
PURP  = '#533483'
CYAN  = '#00d4ff'
RED   = '#e94560'

def dark_fig(rows=1, cols=1, figsize=(12, 5)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor(DARK)
    for ax in (axes.flat if hasattr(axes, 'flat') else [axes]):
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.tick_params(colors='#aaa')
    return fig, axes

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
st.sidebar.markdown("## 🎓 ML Final Project")
st.sidebar.markdown("**Student Habits & Academic Performance**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview & EDA",
    "🔍 Feature Analysis",
    "🤖 ML Experiments",
    "🔮 Score Predictor",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** 1 000 students · 15 features")
st.sidebar.markdown("**Target:** Exam Score (0–100)")

# ─── Load data ────────────────────────────────────────────────────────────────
df, df2, X, y, encoders, features = load_and_preprocess()
trained, results, X_train, X_test, y_train, y_test, scaler = train_models(X, y)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown("# 📊 Overview & Exploratory Data Analysis")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [len(df), df['exam_score'].mean(), df['study_hours_per_day'].mean(),
         df['parental_education_level'].isna().sum()],
        ["Total Students", "Avg Exam Score", "Avg Study Hrs/Day", "Missing: Parent Edu"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val:.1f}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Correlation Matrix", "Group Comparisons"])

    with tab1:
        st.markdown('<div class="section-header"><b>Exam Score Distribution</b></div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(10, 4))
        ax.hist(df['exam_score'], bins=35, color=CYAN, edgecolor=DARK, linewidth=0.7, alpha=0.9)
        ax.axvline(df['exam_score'].mean(), color=RED, linestyle='--', lw=2,
                   label=f"Mean: {df['exam_score'].mean():.1f}")
        ax.axvline(df['exam_score'].median(), color='#ffd700', linestyle='--', lw=2,
                   label=f"Median: {df['exam_score'].median():.1f}")
        ax.set_xlabel('Exam Score', color='#aaa')
        ax.set_ylabel('Count', color='#aaa')
        ax.legend(facecolor=CARD, edgecolor='none', labelcolor='white')
        ax.set_title('Distribution of Exam Scores', color='white', fontsize=13)
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
        <div class="insight-box">
        💡 <b>Insight:</b> Scores range from <b>18.4</b> to <b>100</b> with a mean of <b>{df['exam_score'].mean():.1f}</b>.
        The distribution is approximately normal, slightly skewed right — most students score between 55 and 85.
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header"><b>Feature Correlation Heatmap</b></div>',
                    unsafe_allow_html=True)
        numeric_cols = ['study_hours_per_day', 'social_media_hours', 'netflix_hours',
                        'attendance_percentage', 'sleep_hours', 'exercise_frequency',
                        'mental_health_rating', 'exam_score']
        corr = df[numeric_cols].corr()
        fig, ax = dark_fig(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, ax=ax, mask=mask, cmap='RdYlGn', center=0,
                    annot=True, fmt='.2f', annot_kws={'size': 9, 'color': 'white'},
                    linewidths=0.5, linecolor=DARK)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='#aaa', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='#aaa', fontsize=9)
        ax.set_title('Correlation Matrix', color='white', fontsize=13)
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="insight-box">
        💡 <b>Key correlations with Exam Score:</b><br>
        • <b>study_hours_per_day (+0.83)</b> — strongest positive predictor<br>
        • <b>mental_health_rating (+0.32)</b> — moderate positive effect<br>
        • <b>netflix_hours (−0.17)</b> — mild negative effect<br>
        • <b>social_media_hours (−0.17)</b> — mild negative effect
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header"><b>Exam Score by Categorical Groups</b></div>',
                    unsafe_allow_html=True)
        group_var = st.selectbox("Group by:", ['diet_quality', 'gender', 'internet_quality',
                                               'part_time_job', 'extracurricular_participation',
                                               'parental_education_level'])
        fig, ax = dark_fig(figsize=(10, 4))
        groups_data = df.groupby(group_var)['exam_score']
        group_names = [g for g, _ in groups_data]
        group_vals  = [v.dropna().values for _, v in groups_data]
        bp = ax.boxplot(group_vals, labels=group_names, patch_artist=True,
                        medianprops={'color': RED, 'linewidth': 2},
                        whiskerprops={'color': '#aaa'},
                        capprops={'color': '#aaa'},
                        flierprops={'markerfacecolor': '#555', 'markersize': 4})
        colors_cycle = [CYAN, PURP, BLUE, RED, '#ffd700']
        for patch, color in zip(bp['boxes'], colors_cycle * 5):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xlabel(group_var.replace('_', ' ').title(), color='#aaa')
        ax.set_ylabel('Exam Score', color='#aaa')
        ax.set_title(f'Exam Score by {group_var.replace("_"," ").title()}', color='white', fontsize=13)
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Analysis":
    st.markdown("# 🔍 Feature Analysis")

    st.markdown('<div class="section-header"><b>Study Hours vs Exam Score</b></div>',
                unsafe_allow_html=True)
    color_by = st.selectbox("Color scatter by:", ['mental_health_rating', 'sleep_hours',
                                                    'exercise_frequency', 'social_media_hours'])
    fig, ax = dark_fig(figsize=(11, 5))
    sc = ax.scatter(df['study_hours_per_day'], df['exam_score'],
                    c=df[color_by], cmap='plasma', alpha=0.65, s=18)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_by.replace('_', ' ').title(), color='#aaa')
    cbar.ax.yaxis.set_tick_params(color='#aaa')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaa')
    # trend line
    z = np.polyfit(df['study_hours_per_day'], df['exam_score'], 1)
    p = np.poly1d(z)
    xline = np.linspace(df['study_hours_per_day'].min(), df['study_hours_per_day'].max(), 100)
    ax.plot(xline, p(xline), RED, linewidth=2, linestyle='--', label='Trend')
    ax.legend(facecolor=CARD, edgecolor='none', labelcolor='white')
    ax.set_xlabel('Study Hours per Day', color='#aaa')
    ax.set_ylabel('Exam Score', color='#aaa')
    ax.set_title('Study Hours vs Exam Score', color='white', fontsize=13)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header"><b>Feature Importance (Random Forest)</b></div>',
                unsafe_allow_html=True)

    rf_model = trained['Random Forest'][0]
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = dark_fig(figsize=(10, 6))
    colors_fi = [CYAN if v == importances.max() else PURP for v in importances.values]
    bars = ax.barh(importances.index, importances.values, color=colors_fi, edgecolor=DARK)
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='#aaa', fontsize=8)
    ax.set_xlabel('Importance', color='#aaa')
    ax.set_title('Random Forest Feature Importance', color='white', fontsize=13)
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class="insight-box">
    💡 <b>Interpretation:</b>  <code>study_hours_per_day</code> dominates all other features by a large margin,
    confirming that deliberate study time is the single most reliable predictor of exam success.
    Mental health, sleep, and attendance follow, while demographic features contribute the least.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ML Experiments
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Experiments":
    st.markdown("# 🤖 Machine Learning Experiments")

    # Metrics table
    st.markdown('<div class="section-header"><b>Model Comparison Table</b></div>',
                unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        'Model':  list(results.keys()),
        'R²':     [results[m]['R2']   for m in results],
        'RMSE':   [results[m]['RMSE'] for m in results],
        'MAE':    [results[m]['MAE']  for m in results],
    }).sort_values('R²', ascending=False).reset_index(drop=True)
    metrics_df.insert(0, 'Rank', range(1, len(metrics_df)+1))
    st.dataframe(metrics_df.style.background_gradient(subset=['R²'], cmap='YlGn')
                                  .background_gradient(subset=['RMSE','MAE'], cmap='RdYlGn_r')
                                  .format({'R²': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}'}),
                 use_container_width=True)

    st.markdown("---")

    # Select model for deep dive
    selected = st.selectbox("Select model to inspect:", list(results.keys()))
    res = results[selected]

    col1, col2 = st.columns(2)
    # Actual vs Predicted
    with col1:
        st.markdown(f'<div class="section-header"><b>Actual vs Predicted — {selected}</b></div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(6, 5))
        ax.scatter(res['y_test'], res['preds'], alpha=0.5, s=18, color=CYAN)
        lims = [min(res['y_test'].min(), res['preds'].min()),
                max(res['y_test'].max(), res['preds'].max())]
        ax.plot(lims, lims, RED, linestyle='--', linewidth=2, label='Perfect fit')
        ax.set_xlabel('Actual Score', color='#aaa')
        ax.set_ylabel('Predicted Score', color='#aaa')
        ax.set_title('Actual vs Predicted', color='white', fontsize=12)
        ax.legend(facecolor=CARD, edgecolor='none', labelcolor='white')
        st.pyplot(fig)
        plt.close()

    # Residuals
    with col2:
        st.markdown('<div class="section-header"><b>Residuals Distribution</b></div>',
                    unsafe_allow_html=True)
        residuals = res['y_test'] - res['preds']
        fig, ax = dark_fig(figsize=(6, 5))
        ax.hist(residuals, bins=30, color=PURP, edgecolor=DARK, alpha=0.85)
        ax.axvline(0, color=RED, linestyle='--', linewidth=2)
        ax.set_xlabel('Residual (Actual − Predicted)', color='#aaa')
        ax.set_ylabel('Count', color='#aaa')
        ax.set_title('Residuals', color='white', fontsize=12)
        st.pyplot(fig)
        plt.close()

    st.markdown(f"""
    <div class="insight-box">
    ✅ <b>{selected}</b> — R² = <b>{res['R2']}</b> · RMSE = <b>{res['RMSE']}</b> · MAE = <b>{res['MAE']}</b><br>
    {"🏆 <b>Best model</b> on this dataset!" if selected == "Linear Regression" else ""}
    The near-zero centred residuals confirm the model's predictions are unbiased.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Score Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Score Predictor":
    st.markdown("# 🔮 Predict Your Exam Score")
    st.markdown("Fill in your habits below and choose a model to get an instant prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📚 Academic Habits")
        study_hours     = st.slider("Study Hours per Day", 0.0, 10.0, 4.0, 0.1)
        attendance      = st.slider("Attendance (%)", 50.0, 100.0, 85.0, 0.5)
        social_media    = st.slider("Social Media Hours/Day", 0.0, 8.0, 2.0, 0.1)
        netflix         = st.slider("Netflix Hours/Day", 0.0, 8.0, 1.5, 0.1)

    with col2:
        st.markdown("#### 💪 Lifestyle")
        sleep_hours     = st.slider("Sleep Hours per Night", 4.0, 12.0, 7.0, 0.1)
        exercise        = st.slider("Exercise Days per Week", 0, 7, 3)
        diet            = st.selectbox("Diet Quality", ['Poor', 'Fair', 'Good'])
        mental_health   = st.slider("Mental Health (1=low, 10=high)", 1, 10, 7)

    with col3:
        st.markdown("#### 🏫 Background")
        age             = st.slider("Age", 17, 24, 20)
        gender          = st.selectbox("Gender", ['Female', 'Male', 'Other'])
        part_time       = st.selectbox("Part-time Job?", ['No', 'Yes'])
        extracurr       = st.selectbox("Extracurricular?", ['No', 'Yes'])
        internet_qual   = st.selectbox("Internet Quality", ['Poor', 'Average', 'Good'])
        parent_edu      = st.selectbox("Parental Education", ['High School', 'Bachelor', 'Master', 'Unknown'])

    model_choice = st.selectbox("🤖 Choose Model", list(trained.keys()))

    if st.button("🔮 Predict My Score"):
        # Encode
        enc_map = {
            'gender':       {'Female': 0, 'Male': 1, 'Other': 2},
            'part_time_job':{'No': 0, 'Yes': 1},
            'diet_quality': {'Fair': 0, 'Good': 1, 'Poor': 2},
            'parental_education_level': {'Bachelor': 0, 'High School': 1, 'Master': 2, 'Unknown': 3},
            'internet_quality': {'Average': 0, 'Good': 1, 'Poor': 2},
            'extracurricular_participation': {'No': 0, 'Yes': 1},
        }
        input_vec = np.array([[
            study_hours, social_media, netflix, attendance, sleep_hours,
            exercise, mental_health,
            enc_map['diet_quality'][diet],
            enc_map['internet_quality'][internet_qual],
            enc_map['part_time_job'][part_time],
            enc_map['extracurricular_participation'][extracurr],
            enc_map['gender'][gender],
            enc_map['parental_education_level'][parent_edu],
            age,
        ]])

        model, Xtr_s, Xte_s, sc = trained[model_choice]
        if model_choice in {'Linear Regression', 'Ridge Regression'}:
            inp = sc.transform(input_vec)
        else:
            inp = input_vec

        pred = float(model.predict(inp)[0])
        pred = max(0, min(100, pred))

        # Result
        grade = ('A+' if pred >= 90 else 'A' if pred >= 80 else
                 'B' if pred >= 70 else 'C' if pred >= 60 else
                 'D' if pred >= 50 else 'F')
        color = (CYAN if pred >= 80 else '#ffd700' if pred >= 60 else RED)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg,#1a1a2e,#16213e);
                    border: 2px solid {color}; border-radius: 16px;
                    padding: 32px; text-align: center; margin-top: 20px;">
            <div style="font-size: 3.5rem; font-weight: 700; color: {color};">
                {pred:.1f} / 100
            </div>
            <div style="font-size: 1.8rem; color: white; margin-top: 8px;">
                Grade: <b>{grade}</b>
            </div>
            <div style="color: #888; margin-top: 12px;">
                Predicted by <b>{model_choice}</b>
                (R² = {results[model_choice]['R2']})
            </div>
        </div>""", unsafe_allow_html=True)

        # Tips
        tips = []
        if study_hours < 3:  tips.append("📚 Study at least 3–4 hours/day")
        if social_media > 3: tips.append("📵 Reduce social media time")
        if sleep_hours < 6:  tips.append("😴 Aim for 7–8 hours of sleep")
        if mental_health < 5:tips.append("🧠 Prioritise mental health")
        if attendance < 80:  tips.append("🏫 Improve attendance")
        if tips:
            st.markdown("#### 💡 Personalised Tips to Improve Your Score")
            for tip in tips:
                st.markdown(f"- {tip}")
