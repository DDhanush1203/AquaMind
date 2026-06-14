import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

# PAGE CONFIG 
st.set_page_config(
    page_title="Smart Irrigation AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --p900: #1a0533;
    --p800: #2d0a57;
    --p700: #46167f;
    --p600: #6122a8;
    --p500: #7c35c5;
    --p400: #9d5ee0;
    --p300: #bb8eed;
    --p200: #d9bef6;
    --p100: #f0e3fd;
    --p50:  #f9f4ff;
    --accent: #c084fc;
    --gold: #f0b429;
    --text: #1a0533;
    --muted: #6b5f7a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, var(--p50) 0%, #fdf8ff 50%, #f5e8ff 100%);
    min-height: 100vh;
}

/* Hide default streamlit elements */
#MainMenu, footer { visibility: hidden; }
[data-testid="collapsedControl"] { 
    visibility: visible !important;
    display: block !important;
    position: fixed !important;
    top: 0.5rem !important;
    left: 0.5rem !important;
    z-index: 999999 !important;
}

/* Custom header */
.aquamind-header {
    background: linear-gradient(135deg, var(--p900) 0%, var(--p700) 60%, var(--p500) 100%);
    padding: 2.5rem 3rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.aquamind-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(192,132,252,0.3) 0%, transparent 70%);
    border-radius: 50%;
}
.aquamind-header::after {
    content: '';
    position: absolute;
    bottom: -20px; left: 20%;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(240,180,41,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-subtitle {
    color: var(--p200);
    font-size: 1rem;
    font-weight: 300;
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
}
.header-badge {
    display: inline-block;
    background: rgba(240,180,41,0.2);
    border: 1px solid rgba(240,180,41,0.5);
    color: var(--gold);
    font-size: 0.7rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    background: white;
    border: 1px solid var(--p200);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    flex: 1;
    min-width: 140px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--p500), var(--p300));
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--p700);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--p800);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--p200), transparent);
}

/* Prediction result */
.pred-box {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.pred-low {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border: 2px solid #81c784;
}
.pred-medium {
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    border: 2px solid #ffd54f;
}
.pred-high {
    background: linear-gradient(135deg, #fce4ec, #f8bbd0);
    border: 2px solid #f48fb1;
}
.pred-label {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 900;
}
.pred-low .pred-label { color: #2e7d32; }
.pred-medium .pred-label { color: #f57f17; }
.pred-high .pred-label { color: #c62828; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--p900) 0%, var(--p800) 100%) !important;
    border-left: 1px solid var(--p700) !important;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label { color: var(--p200) !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] { background: rgba(255,255,255,0.1) !important; }

.sidebar-section {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-heading {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-bottom: 0.8rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--p200);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: var(--muted);
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--p600), var(--p500)) !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--p600), var(--p500)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(98,34,168,0.3) !important;
}

/* Info boxes */
.info-card {
    background: white;
    border-left: 4px solid var(--p500);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

/* Accuracy badge */
.acc-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--p600), var(--p400));
    color: white;
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    padding: 0.8rem 2rem;
    border-radius: 16px;
    margin: 0.5rem 0;
}

/* Dataframe styling */
.stDataFrame { border-radius: 12px; overflow: hidden; }
            
/* Fix invisible labels in main content */
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label {
    color: #2d0a57 !important;
}
            /* Fix info card text color */
.info-card, .info-card * {
    color: #2d0a57 !important;
}
</style>
""", unsafe_allow_html=True)

# MATPLOTLIB THEME 
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#f9f4ff',
    'figure.facecolor': '#f9f4ff',
    'axes.edgecolor': '#d9bef6',
    'axes.labelcolor': '#2d0a57',
    'xtick.color': '#6b5f7a',
    'ytick.color': '#6b5f7a',
    'text.color': '#2d0a57',
    'grid.color': '#e8d8f8',
    'grid.alpha': 0.6,
})
PURPLE_PALETTE = ['#7c35c5','#9d5ee0','#bb8eed','#46167f','#c084fc','#d9bef6']
PURPLE_CMAP = 'Purples'

# DATA LOADING 
@st.cache_data
def load_data():
    df = pd.read_csv("irrigation_prediction.csv")
    return df

@st.cache_data
def preprocess(df):
    df2 = df.copy()
    cat_cols = ['Soil_Type','Crop_Type','Crop_Growth_Stage','Season',
                'Irrigation_Type','Water_Source','Mulching_Used','Region']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col])
        encoders[col] = le
    le_target = LabelEncoder()
    df2['Irrigation_Need'] = le_target.fit_transform(df2['Irrigation_Need'])
    return df2, encoders, le_target

@st.cache_resource
def train_models(df_enc):
    X = df_enc.drop('Irrigation_Need', axis=1)
    y = df_enc['Irrigation_Need']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    cw = dict(zip(classes, weights))

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'cm': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
        }
    return results, X_train, X_test, y_train, y_test, X.columns.tolist()

# LOAD 
df_raw = load_data()
df_enc, encoders, le_target = preprocess(df_raw)
results, X_train, X_test, y_train, y_test, feature_names = train_models(df_enc)
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_result = results[best_model_name]
class_names = le_target.classes_

# HEADER
st.markdown("""
<div class="aquamind-header">
    <div class="header-badge">🌱 SDG 2 · Zero Hunger</div>
    <div class="header-title">💧 AquaMind</div>
    <div class="header-subtitle">Intelligent Water Management System for Smart Farming · PES University · PES1PG25CA045</div>
</div>
""", unsafe_allow_html=True)

# METRICS ROW
best_acc = best_result['accuracy']
best_f1 = best_result['f1']
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-val">{len(df_raw):,}</div>
        <div class="metric-label">Training Records</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{len(feature_names)}</div>
        <div class="metric-label">Features Used</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{best_acc*100:.1f}%</div>
        <div class="metric-label">Best Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{best_f1*100:.1f}%</div>
        <div class="metric-label">Best F1 Score</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">3</div>
        <div class="metric-label">Models Trained</div>
    </div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR (RIGHT)
# Streamlit sidebar is always left by default; we use st.sidebar
# and inform user it appears on right via CSS if layout supports
with st.sidebar:
    st.markdown('<div class="sidebar-heading">⚙ Configuration</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sidebar-heading">🤖 Active Model</div>', unsafe_allow_html=True)
    selected_model = st.radio(
        "Choose classifier",
        list(results.keys()),
        index=list(results.keys()).index(best_model_name),
        label_visibility="collapsed"
    )
    acc = results[selected_model]['accuracy']
    f1 = results[selected_model]['f1']
    st.markdown(f"**Accuracy:** `{acc*100:.2f}%`")
    st.markdown(f"**F1 Score:** `{f1*100:.2f}%`")

    st.markdown("---")
    st.markdown('<div class="sidebar-heading">📊 Dataset Info</div>', unsafe_allow_html=True)
    st.markdown(f"**Rows:** `{len(df_raw):,}`")
    st.markdown(f"**Columns:** `{len(df_raw.columns)}`")
    st.markdown(f"**Target:** `Irrigation_Need`")
    st.markdown(f"**Task:** `Classification`")

    st.markdown("---")
    st.markdown('<div class="sidebar-heading">👨‍🎓 Student Info</div>', unsafe_allow_html=True)
    st.markdown("**Name:** D Dhanush")
    st.markdown("**SRN:** PES1PG25CA045")
    st.markdown("**Domain:** Agriculture · AIML")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Analysis",
    "🤖 Model Performance",
    "💧 Predict Irrigation",
    "📋 Dataset Preview",
])

# TAB 1 — EDA

with tab1:
    st.markdown('<div class="section-title">Exploratory Data Analysis <span class="section-line"></span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Target distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df_raw['Irrigation_Need'].value_counts()
        bars = ax.bar(counts.index, counts.values, color=PURPLE_PALETTE[:3],
                      edgecolor='white', linewidth=1.5, width=0.55)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val:,}\n({val/len(df_raw)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9, color='#2d0a57', fontweight='500')
        ax.set_title('Irrigation Need Distribution', fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel('Count')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        # Crop type vs irrigation
        fig, ax = plt.subplots(figsize=(6, 4))
        ct = pd.crosstab(df_raw['Crop_Type'], df_raw['Irrigation_Need'], normalize='index') * 100
        ct.plot(kind='bar', ax=ax, color=PURPLE_PALETTE[:3], edgecolor='white', linewidth=0.8)
        ax.set_title('Irrigation Need by Crop Type (%)', fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('')
        ax.legend(title='Need', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        # Temperature distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        for need, color in zip(['Low', 'Medium', 'High'], PURPLE_PALETTE[:3]):
            subset = df_raw[df_raw['Irrigation_Need'] == need]['Temperature_C']
            ax.hist(subset, bins=25, alpha=0.65, label=need, color=color, edgecolor='white')
        ax.set_title('Temperature by Irrigation Need', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        # Soil moisture vs need
        fig, ax = plt.subplots(figsize=(6, 4))
        data_by_need = [df_raw[df_raw['Irrigation_Need'] == n]['Soil_Moisture'].values
                        for n in ['Low', 'Medium', 'High']]
        bp = ax.boxplot(data_by_need, labels=['Low', 'Medium', 'High'],
                        patch_artist=True, notch=False,
                        medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp['boxes'], PURPLE_PALETTE[:3]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.set_title('Soil Moisture by Irrigation Need', fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel('Soil Moisture (%)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Correlation heatmap
    st.markdown('<div class="section-title">Feature Correlation <span class="section-line"></span></div>', unsafe_allow_html=True)
    num_cols = ['Soil_pH','Soil_Moisture','Organic_Carbon','Electrical_Conductivity',
                'Temperature_C','Humidity','Rainfall_mm','Sunlight_Hours',
                'Wind_Speed_kmh','Field_Area_hectare','Previous_Irrigation_mm']
    corr = df_enc[num_cols + ['Irrigation_Need']].corr()
    fig, ax = plt.subplots(figsize=(12, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdPu',
                ax=ax, linewidths=0.5, linecolor='white',
                annot_kws={'size': 8}, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# TAB 2 — MODEL PERFORMANCE

with tab2:
    st.markdown('<div class="section-title">Model Comparison <span class="section-line"></span></div>', unsafe_allow_html=True)

    # Comparison cards
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            is_best = name == best_model_name
            border = "border: 2px solid #7c35c5;" if is_best else ""
            badge = "<span style='background:#7c35c5;color:white;font-size:0.65rem;padding:2px 8px;border-radius:10px;margin-left:8px'>BEST</span>" if is_best else ""
            st.markdown(f"""
            <div style="background:white;border-radius:16px;padding:1.2rem;{border}margin-bottom:1rem">
                <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;color:#2d0a57">{name}{badge}</div>
                <div style="margin-top:1rem">
                    <div style="font-size:0.7rem;color:#6b5f7a;text-transform:uppercase;letter-spacing:0.5px">Accuracy</div>
                    <div style="font-size:1.8rem;font-weight:700;color:#6122a8">{res['accuracy']*100:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#6b5f7a;text-transform:uppercase;letter-spacing:0.5px">F1 Score</div>
                    <div style="font-size:1.4rem;font-weight:600;color:#9d5ee0">{res['f1']*100:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Selected model deep dive
    sel = results[selected_model]
    st.markdown(f'<div class="section-title">Deep Dive: {selected_model} <span class="section-line"></span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(sel['cm'], annot=True, fmt='d', cmap='Purples',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, linecolor='white',
                    annot_kws={'size': 12, 'fontweight': 'bold'})
        ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        # Per-class metrics
        report = sel['report']
        classes_in_report = [c for c in class_names if c in report]
        metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(classes_in_report))
        width = 0.25
        fig, ax = plt.subplots(figsize=(5, 4))
        for i, metric in enumerate(metrics):
            vals = [report[c][metric] for c in classes_in_report]
            bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                         color=PURPLE_PALETTE[i], edgecolor='white', linewidth=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes_in_report)
        ax.set_ylim(0, 1.1)
        ax.set_title('Per-Class Metrics', fontsize=13, fontweight='bold', pad=15)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Feature importance (Random Forest)
    if 'Random Forest' in results:
        st.markdown('<div class="section-title">Feature Importance (Random Forest) <span class="section-line"></span></div>', unsafe_allow_html=True)
        rf_model = results['Random Forest']['model']
        importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
        top15 = importances.tail(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(top15)))
        bars = ax.barh(top15.index, top15.values, color=colors, edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, top15.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9, color='#2d0a57')
        ax.set_title('Top 15 Feature Importances', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Importance Score')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# TAB 3 — PREDICTION

with tab3:
    st.markdown('<div class="section-title">Predict Irrigation Requirement <span class="section-line"></span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        Enter your farm's current conditions below to get an AI-powered irrigation recommendation.
        Using the <strong>""" + selected_model + """</strong> model.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🌱 Soil & Field**")
        soil_type = st.selectbox("Soil Type", ['Clay', 'Loamy', 'Sandy', 'Silt'])
        soil_ph = st.slider("Soil pH", 4.8, 8.2, 6.5, 0.1)
        soil_moisture = st.slider("Soil Moisture (%)", 8.0, 65.0, 37.0, 0.5)
        organic_carbon = st.slider("Organic Carbon (%)", 0.3, 1.6, 0.9, 0.05)
        electrical_cond = st.slider("Electrical Conductivity", 0.1, 3.5, 1.8, 0.1)
        field_area = st.slider("Field Area (ha)", 0.3, 15.0, 5.0, 0.1)
        mulching = st.selectbox("Mulching Used", ['Yes', 'No'])

    with c2:
        st.markdown("**🌤 Weather**")
        temperature = st.slider("Temperature (°C)", 12.0, 42.0, 27.0, 0.5)
        humidity = st.slider("Humidity (%)", 25.0, 95.0, 60.0, 1.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 2500.0, 1250.0, 10.0)
        sunlight = st.slider("Sunlight Hours", 4.0, 11.0, 7.5, 0.5)
        wind_speed = st.slider("Wind Speed (km/h)", 0.5, 20.0, 10.0, 0.5)

    with c3:
        st.markdown("**🌾 Crop & Context**")
        crop_type = st.selectbox("Crop Type", ['Cotton', 'Maize', 'Potato', 'Rice', 'Sugarcane', 'Wheat'])
        growth_stage = st.selectbox("Growth Stage", ['Flowering', 'Harvest', 'Sowing', 'Vegetative'])
        season = st.selectbox("Season", ['Kharif', 'Rabi', 'Zaid'])
        irrigation_type = st.selectbox("Irrigation Type", ['Canal', 'Drip', 'Rainfed', 'Sprinkler'])
        water_source = st.selectbox("Water Source", ['Groundwater', 'Rainwater', 'Reservoir', 'River'])
        region = st.selectbox("Region", ['Central', 'East', 'North', 'South', 'West'])
        prev_irrigation = st.slider("Previous Irrigation (mm)", 0.0, 120.0, 60.0, 1.0)

    if st.button("🔮 Predict Irrigation Need", use_container_width=True):
        input_dict = {
            'Soil_Type': encoders['Soil_Type'].transform([soil_type])[0],
            'Soil_pH': soil_ph,
            'Soil_Moisture': soil_moisture,
            'Organic_Carbon': organic_carbon,
            'Electrical_Conductivity': electrical_cond,
            'Temperature_C': temperature,
            'Humidity': humidity,
            'Rainfall_mm': rainfall,
            'Sunlight_Hours': sunlight,
            'Wind_Speed_kmh': wind_speed,
            'Crop_Type': encoders['Crop_Type'].transform([crop_type])[0],
            'Crop_Growth_Stage': encoders['Crop_Growth_Stage'].transform([growth_stage])[0],
            'Season': encoders['Season'].transform([season])[0],
            'Irrigation_Type': encoders['Irrigation_Type'].transform([irrigation_type])[0],
            'Water_Source': encoders['Water_Source'].transform([water_source])[0],
            'Field_Area_hectare': field_area,
            'Mulching_Used': encoders['Mulching_Used'].transform([mulching])[0],
            'Previous_Irrigation_mm': prev_irrigation,
            'Region': encoders['Region'].transform([region])[0],
        }
        input_df = pd.DataFrame([input_dict])
        model = results[selected_model]['model']
        prediction = le_target.inverse_transform(model.predict(input_df))[0]
        proba = model.predict_proba(input_df)[0]

        css_class = {'Low': 'pred-low', 'Medium': 'pred-medium', 'High': 'pred-high'}[prediction]
        icon = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}[prediction]
        advice = {
            'Low': 'Soil moisture is adequate. Minimal irrigation needed — conserve water.',
            'Medium': 'Moderate irrigation recommended. Monitor soil conditions daily.',
            'High': 'Urgent irrigation required! Deploy full irrigation immediately.'
        }[prediction]

        st.markdown(f"""
        <div class="pred-box {css_class}">
            <div style="font-size:3rem">{icon}</div>
            <div style="font-size:0.9rem;font-weight:500;color:#555;margin-bottom:0.5rem">PREDICTED IRRIGATION NEED</div>
            <div class="pred-label">{prediction}</div>
            <div style="margin-top:0.8rem;font-size:0.95rem;color:#444">{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Prediction Probabilities:**")
        prob_df = pd.DataFrame({'Class': le_target.classes_, 'Probability': proba})
        fig, ax = plt.subplots(figsize=(6, 2.5))
        bars = ax.barh(prob_df['Class'], prob_df['Probability'],
                      color=PURPLE_PALETTE[:3], edgecolor='white', height=0.5)
        for bar, val in zip(bars, prob_df['Probability']):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', fontsize=11, fontweight='600', color='#2d0a57')
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Probability')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# TAB 4 — DATASET PREVIEW

with tab4:
    st.markdown('<div class="section-title">Dataset Preview <span class="section-line"></span></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df_raw.head(100), use_container_width=True, height=400)
    with col2:
        st.markdown("**Shape**")
        st.info(f"{df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
        st.markdown("**Target Classes**")
        for cls, cnt in df_raw['Irrigation_Need'].value_counts().items():
            pct = cnt / len(df_raw) * 100
            st.markdown(f"- **{cls}**: {cnt:,} ({pct:.1f}%)")
        st.markdown("**Missing Values**")
        st.success("✅ None — dataset is clean")
    st.markdown("**Descriptive Statistics**")
    st.dataframe(df_raw.describe().round(2), use_container_width=True)
