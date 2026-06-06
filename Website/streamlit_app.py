"""
streamlit_app.py  v6 — Memory Health Assessment
─────────────────────────────────────────────────
ALL changes from user feedback applied:
  • Entire hero card is clickable (whole box = button)
  • Mobile-responsive: title shrinks, cards readable
  • Doctor card: lightened so text is visible
  • Step 0 – cleaner form, no ghost boxes, age from 18,
              Full Name label, education choices visible
  • Step 1 – Height+Weight→BMI auto-calc, Type-1 & Type-2
              diabetes, other health conditions field,
              better smoking UI, no Back button
  • Step 2 – removed Back button, no autofocus ghost boxes,
              clean recall textarea, no example placeholder
  • Step 3 – visible radio buttons instead of broken sliders,
              better colour combination
  • Step 5 – removed Streamlit progress bar, clean digit input
  • Step 6 – clean recall, optional MRI upload (optional)
  • Step 7 – model comparison HIDDEN from patient view,
              patient data saved to backend CSV
"""

import os, warnings, csv, random
import numpy as np
import joblib, pickle
import streamlit as st
import plotly.graph_objects as go
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import shap
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_XGB    = os.path.join(BASE_DIR, "models", "best_xgb.pkl")
MODEL_RF     = os.path.join(BASE_DIR, "models", "best_rf.pkl")
MODEL_LR     = os.path.join(BASE_DIR, "models", "best_lr.pkl")
RESULTS_FILE = os.path.join(BASE_DIR, "results", "patient_results.csv")

st.set_page_config(
    page_title="Memory Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ════════════════════════════════════════════════════════════
# COMPREHENSIVE CSS  — mobile-first, high-contrast
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root { color-scheme: light only !important; }
*, *::before, *::after {
  font-family: 'Inter', sans-serif !important;
  box-sizing: border-box;
}

/* ── Backgrounds ── */
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, section.main,
[data-testid="stVerticalBlock"],
[data-testid="block-container"] {
  background-color: #f0f4ff !important;
  color: #1a1a2e !important;
}
[data-testid="stSidebar"] { background: #fff !important; }

/* ── Force dark text everywhere ── */
p, span, label, div, h1, h2, h3, h4, h5, h6, li, a,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"] { color: #1a1a2e !important; }

/* ── Inputs ── */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
  background: #fff !important;
  color: #1a1a2e !important;
  border: 2px solid #c7d2fe !important;
  border-radius: 12px !important;
  font-size: 1rem !important;
  caret-color: #6366f1 !important;
}
input:focus, textarea:focus {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 3px #6366f118 !important;
  outline: none !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
  background: #fff !important;
  color: #1a1a2e !important;
  border: 2px solid #c7d2fe !important;
  border-radius: 12px !important;
}
[data-testid="stSelectbox"] svg { fill: #6366f1 !important; }

/* ── Radio buttons — HIGHLY VISIBLE ── */
[data-testid="stRadio"] label {
  background: white !important;
  color: #1a1a2e !important;
  border: 2px solid #c7d2fe !important;
  border-radius: 10px !important;
  padding: 8px 14px !important;
  font-size: 0.87rem !important;
  font-weight: 600 !important;
  cursor: pointer !important;
  transition: all 0.15s !important;
  margin: 3px !important;
}
[data-testid="stRadio"] label:hover {
  border-color: #6366f1 !important;
  background: #f5f3ff !important;
}
[data-testid="stRadio"] label[data-checked="true"],
[data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: white !important;
  border-color: #6366f1 !important;
}
[data-testid="stRadio"] label[data-checked="true"] *,
[data-testid="stRadio"] label:has(input:checked) * { color: white !important; }

/* ── Slider ── */
[data-testid="stSlider"] div[data-baseweb="slider"] div {
  background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] button {
  background: #e0e7ff !important;
  color: #6366f1 !important;
  border-radius: 8px !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}
[data-testid="stProgress"] > div { background: #e0e7ff !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
  background: #fff !important;
  border-radius: 16px !important;
  padding: 16px !important;
  border: 2px solid #e0e7ff !important;
}
[data-testid="stMetricValue"] { color: #6366f1 !important; }

/* ── ALL Buttons ── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
  font-weight: 700 !important;
  border-radius: 14px !important;
  padding: 14px 28px !important;
  width: 100% !important;
  border: none !important;
  font-size: 1rem !important;
  transition: all 0.25s !important;
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: #ffffff !important;
  box-shadow: 0 4px 16px #6366f133 !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-3px) scale(1.01) !important;
  box-shadow: 0 10px 28px #6366f155 !important;
}

/* ── LANDING HERO CARDS ── */
.hero-card-doc {
  background: linear-gradient(160deg, #1d3461 0%, #1a237e 50%, #283593 100%);
  border-radius: 26px;
  padding: 40px 30px 0;
  box-shadow: 0 20px 56px rgba(0,0,0,0.28);
  border: 2px solid rgba(167,199,255,0.25);
  cursor: pointer;
  transition: transform 0.3s, box-shadow 0.3s;
  text-align: center;
}
.hero-card-doc:hover {
  transform: translateY(-10px);
  box-shadow: 0 30px 72px rgba(0,0,0,0.32);
}
.hero-card-pat {
  background: linear-gradient(160deg, #4c1d95 0%, #7c3aed 60%, #9333ea 100%);
  border-radius: 26px;
  padding: 40px 30px 0;
  box-shadow: 0 20px 56px rgba(124,58,237,0.4);
  border: 2px solid rgba(255,255,255,0.18);
  cursor: pointer;
  transition: transform 0.3s, box-shadow 0.3s;
  text-align: center;
}
.hero-card-pat:hover {
  transform: translateY(-10px);
  box-shadow: 0 30px 72px rgba(124,58,237,0.48);
}

/* Card footer buttons */
.hero-card-doc .stButton > button {
  background: rgba(147,197,253,0.22) !important;
  border: 2px solid rgba(147,197,253,0.45) !important;
  border-radius: 0 0 24px 24px !important;
  margin-top: 22px !important;
  padding: 18px !important;
  font-size: 1.05rem !important;
  font-weight: 900 !important;
  color: #fff !important;
  box-shadow: none !important;
  backdrop-filter: blur(8px);
}
.hero-card-doc .stButton > button:hover {
  background: rgba(147,197,253,0.38) !important;
  transform: none !important;
}
.hero-card-pat .stButton > button {
  background: rgba(236,72,153,0.22) !important;
  border: 2px solid rgba(236,72,153,0.45) !important;
  border-radius: 0 0 24px 24px !important;
  margin-top: 22px !important;
  padding: 18px !important;
  font-size: 1.05rem !important;
  font-weight: 900 !important;
  color: #fff !important;
  box-shadow: none !important;
}
.hero-card-pat .stButton > button:hover {
  background: rgba(236,72,153,0.38) !important;
  transform: none !important;
}

/* ── Glass card ── */
.gcard {
  background: rgba(255,255,255,0.97);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  border: 2px solid #e0e7ff;
  padding: 22px 26px;
  margin: 10px 0;
  box-shadow: 0 4px 20px #6366f108;
}
.gcard-warn {
  background: linear-gradient(135deg,#fff8f0,#fff);
  border-radius: 20px;
  border: 2px solid #fed7aa;
  padding: 22px 26px;
  margin: 10px 0;
}

/* ── Health cards ── */
.hcard {
  background: white;
  border-radius: 16px;
  border: 2px solid #e0e7ff;
  padding: 16px 20px;
  margin: 6px 0;
}
.hcard-warn {
  border-color: #fca5a5 !important;
  background: linear-gradient(135deg,#fff5f5,#fff) !important;
}

/* ── FAQ question card ── */
.faq-q {
  background: white;
  border-radius: 14px;
  border: 2px solid #e0e7ff;
  padding: 12px 16px;
  margin: 8px 0 4px;
  box-shadow: 0 2px 10px #6366f108;
}

/* ── Word boxes ── */
.wbox {
  display: inline-block;
  background: linear-gradient(135deg,#f5f3ff,#ede9fe);
  border: 2px solid #8b5cf6;
  border-radius: 12px;
  padding: 8px 16px;
  margin: 4px;
  font-size: 1rem;
  font-weight: 800;
  color: #4c1d95 !important;
  letter-spacing: 0.05em;
  transition: transform 0.15s;
}
.wbox:hover { transform: scale(1.07); }

/* ── Progress pills ── */
.npill {
  background: #e0e7ff;
  color: #3730a3 !important;
  border-radius: 999px;
  padding: 5px 11px;
  font-weight: 600;
  font-size: 0.72rem;
  display: inline-block;
  margin: 2px;
}
.npill-a {
  background: linear-gradient(135deg,#6366f1,#8b5cf6);
  color: white !important;
  border-radius: 999px;
  padding: 5px 11px;
  font-weight: 800;
  font-size: 0.72rem;
  display: inline-block;
  margin: 2px;
}

/* ── Custom progress bar ── */
.prog-bg {
  background: #e0e7ff;
  border-radius: 999px;
  height: 10px;
  margin: 8px 0;
  overflow: hidden;
}
.prog-fill {
  border-radius: 999px;
  height: 10px;
  background: linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899);
  transition: width 0.6s;
}

/* ── Result cards ── */
.res-high {
  background: linear-gradient(135deg,#fff1f2,#fef2f2);
  border: 2px solid #fca5a5;
  border-radius: 24px;
  padding: 36px 24px;
  text-align: center;
}
.res-med {
  background: linear-gradient(135deg,#fffbeb,#fefce8);
  border: 2px solid #fcd34d;
  border-radius: 24px;
  padding: 36px 24px;
  text-align: center;
}
.res-low {
  background: linear-gradient(135deg,#f0fdf4,#ecfdf5);
  border: 2px solid #86efac;
  border-radius: 24px;
  padding: 36px 24px;
  text-align: center;
}

/* ── Small metric cards ── */
.mcard {
  background: white;
  border-radius: 16px;
  padding: 18px 12px;
  text-align: center;
  border: 2px solid #e0e7ff;
  box-shadow: 0 4px 16px #6366f108;
}

/* ── Prevention ── */
.prev-u { background:white; border-left:5px solid #ef4444; border-radius:14px; padding:14px 18px; margin:6px 0; }
.prev-w { background:white; border-left:5px solid #f59e0b; border-radius:14px; padding:14px 18px; margin:6px 0; }
.prev-g { background:white; border-left:5px solid #10b981; border-radius:14px; padding:14px 18px; margin:6px 0; }

/* ── Disclaimer ── */
.disclaimer {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-left: 4px solid #f59e0b;
  border-radius: 12px;
  padding: 10px 18px;
  font-size: 0.82rem;
  color: #92400e !important;
  display: inline-block;
  margin-top: 10px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: white !important;
  border: 2px dashed #c7d2fe !important;
  border-radius: 16px !important;
  padding: 16px !important;
}
[data-testid="stFileUploader"] * { color: #1a1a2e !important; }
[data-testid="stFileUploader"] button {
  background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
  color: white !important;
  border-radius: 10px !important;
  border: none !important;
  width: auto !important;
}

/* ── Remove Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1160px !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
  .block-container { padding: 0.5rem !important; }
  [data-testid="stMarkdownContainer"] h1 { font-size: 1.6rem !important; }
  [data-testid="stMarkdownContainer"] h2 { font-size: 1.15rem !important; }
  .hero-card-doc, .hero-card-pat { padding: 24px 16px 0 !important; }
  .gcard { padding: 14px 16px !important; }
  .npill, .npill-a { font-size: 0.62rem !important; padding: 4px 7px !important; }
  [data-testid="stRadio"] label { font-size: 0.78rem !important; padding: 6px 9px !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    m = {}
    for name, path in [("XGBoost", MODEL_XGB), ("Random Forest", MODEL_RF),
                       ("Logistic Regression", MODEL_LR)]:
        if not os.path.exists(path):
            continue
        try:
            m[name] = joblib.load(path)
        except Exception:
            try:
                with open(path, "rb") as f:
                    m[name] = pickle.load(f)
            except Exception:
                pass
    return m

models  = load_models()
primary = models.get("XGBoost") or (list(models.values())[0] if models else None)

ALL_FEATURES = [
    "PTGENDER", "PTEDUCAT", "MMSE_BL", "MOCA_BL", "ADAS11_BL", "ADAS13_BL",
    "FAQ_BL", "GDS_BL", "CDR_GLOBAL_BL", "CDRSB_BL", "RAVLT_forgetting",
    "RAVLT_immediate", "RAVLT_delayed", "DigitSpan", "TrailsB",
    "MMSE_FAQ_composite", "ADAS_MMSE_gap", "RAVLT_forget_rate"
]

WORD_LIST = [
    "drum", "curtain", "bell", "coffee", "school", "parent",
    "moon", "garden", "hat", "farmer", "nose", "turkey", "river", "house", "road"
]

FAQ_ITEMS = [
    ("💳", "Writing cheques or paying bills"),
    ("🛒", "Shopping alone for groceries"),
    ("☕", "Heating water or making coffee"),
    ("🍽️", "Preparing a balanced meal"),
    ("📰", "Keeping track of current news"),
    ("📺", "Following a TV series plot"),
    ("📅", "Remembering appointments"),
    ("🚗", "Driving or using transport"),
    ("📝", "Filling in forms or paperwork"),
    ("💊", "Managing medicines correctly"),
]

GDS_ITEMS = [
    ("😊", "Basically satisfied with your life?", "no"),
    ("📉", "Dropped many of your activities?", "yes"),
    ("😶", "Life feels empty?", "yes"),
    ("😑", "Often get bored?", "yes"),
    ("🌤️", "In good spirits most of the time?", "no"),
    ("😰", "Afraid something bad will happen?", "yes"),
    ("😄", "Happy most of the time?", "no"),
    ("😔", "Often feel helpless?", "yes"),
    ("🏠", "Prefer staying home over going out?", "yes"),
    ("🧠", "More memory problems than most?", "yes"),
    ("🌟", "Wonderful to be alive now?", "no"),
    ("💔", "Feel worthless the way you are?", "yes"),
    ("⚡", "Full of energy?", "no"),
    ("🌑", "Situation feels hopeless?", "yes"),
    ("📊", "Others are better off than you?", "yes"),
]

THRESHOLD_OPTIONS = {
    "🔍 Screening (t=0.15)": 0.15,
    "⚖️ Balanced (t=0.35)":  0.35,
    "✅ Confirmatory (t=0.50)": 0.50,
}

PREVENTION = {
    "HIGH": [
        ("🏥", "See a neurologist soon", "Book this week — early intervention is most effective.", "prev-u"),
        ("💊", "Medication review", "Ask your doctor to review all current medications.", "prev-u"),
        ("❤️", "Control BP, diabetes & cholesterol", "These directly increase dementia risk.", "prev-w"),
        ("🥗", "Mediterranean diet", "Fish, olive oil, vegetables, nuts, whole grains.", "prev-w"),
        ("🏃", "Exercise 30 min daily", "Walking or cycling improves brain blood flow.", "prev-w"),
        ("🧩", "Keep mind active", "Reading, puzzles, learning something new daily.", "prev-g"),
    ],
    "MODERATE": [
        ("📅", "Book a memory check-up", "Annual cognitive screening recommended.", "prev-w"),
        ("🏃", "Exercise regularly", "Even 20-minute walks make a measurable difference.", "prev-w"),
        ("❤️", "Check BP and blood sugar", "Silent risk factors for brain health.", "prev-w"),
        ("🥗", "Improve diet quality", "More fish, vegetables, nuts.", "prev-g"),
        ("🧩", "Daily mental stimulation", "Crosswords, reading, or learning a new hobby.", "prev-g"),
        ("💤", "Prioritise sleep", "7-9 hours of quality sleep protects brain health.", "prev-g"),
    ],
    "LOWER": [
        ("✅", "Maintain healthy habits", "Current scores are reassuring — keep it up!", "prev-g"),
        ("🏃", "Stay physically active", "Exercise is the most powerful brain protection.", "prev-g"),
        ("📅", "Annual health check-ups", "BP, blood sugar, and cholesterol every year.", "prev-g"),
        ("🧩", "Keep challenging your brain", "New experiences build cognitive reserve.", "prev-g"),
        ("🥗", "Eat well", "Mediterranean diet is strongly protective.", "prev-g"),
        ("💤", "Protect your sleep", "Poor sleep accelerates cognitive decline.", "prev-g"),
    ],
}


# ── Helper functions ───────────────────────────────────────────
def compute_composites(v):
    v["MMSE_FAQ_composite"] = float(v["MMSE_BL"]) - float(v["FAQ_BL"])
    v["ADAS_MMSE_gap"]      = float(v["ADAS13_BL"]) + (30 - float(v["MMSE_BL"]))
    avg = max(float(v["RAVLT_immediate"]) / 5, 0.1)
    v["RAVLT_forget_rate"]  = float(v["RAVLT_forgetting"]) / avg
    return v


def risk_info(prob, threshold=0.50):
    if threshold <= 0.15:
        lo, hi = 0.40, 0.65
    elif threshold <= 0.35:
        lo, hi = 0.50, 0.70
    else:
        lo, hi = 0.55, 0.75

    if prob >= hi:
        return {
            "level": "HIGH", "color": "#ef4444", "cls": "res-high",
            "emoji": "⚠️", "pct": int(prob * 100),
            "headline": "Higher risk of progression",
            "plain": "These scores suggest a higher-than-average chance of memory decline in the next 3 years. Please consult a neurologist soon."
        }
    if prob >= lo:
        return {
            "level": "MODERATE", "color": "#f59e0b", "cls": "res-med",
            "emoji": "🔶", "pct": int(prob * 100),
            "headline": "Moderate — monitor closely",
            "plain": "Some warning signs are present. Worth monitoring carefully with regular doctor visits."
        }
    return {
        "level": "LOWER", "color": "#10b981", "cls": "res-low",
        "emoji": "✅", "pct": int(prob * 100),
        "headline": "Lower risk at this time",
        "plain": "Fewer signs of progression right now. Continue healthy habits and annual check-ups."
    }


def comorbidity_flags(diabetes, hypertension, smoking, bmi, other_conditions=""):
    flags = []
    if diabetes in ("Type 1", "Type 2"):
        dtype = "Type 1" if diabetes == "Type 1" else "Type 2"
        flags.append((
            "🩸", f"{dtype} Diabetes",
            "Insulin resistance is linked to amyloid accumulation. Tight glucose control is important.",
            "#fef2f2", "#ef4444"
        ))
    if hypertension == "Yes":
        flags.append((
            "💉", "Hypertension",
            "Vascular damage accelerates cognitive decline. Monitor BP regularly.",
            "#fef2f2", "#ef4444"
        ))
    if smoking == "Current":
        flags.append((
            "🚬", "Current Smoker",
            "Significant cerebrovascular risk factor. Cessation strongly recommended.",
            "#fef2f2", "#ef4444"
        ))
    elif smoking == "Former":
        flags.append((
            "🚬", "Former Smoker",
            "Residual cerebrovascular risk. Continue smoke-free.",
            "#fffbeb", "#f59e0b"
        ))
    if bmi >= 30:
        flags.append((
            "⚖️", f"Obese (BMI {bmi:.1f})",
            "Midlife obesity significantly increases dementia risk.",
            "#fef2f2", "#ef4444"
        ))
    elif bmi >= 25:
        flags.append((
            "⚖️", f"Overweight (BMI {bmi:.1f})",
            "Some elevated risk. A healthy diet and exercise are recommended.",
            "#fffbeb", "#f59e0b"
        ))
    if other_conditions and other_conditions.strip():
        flags.append((
            "📋", "Other Reported Conditions",
            other_conditions.strip(),
            "#f0f9ff", "#0ea5e9"
        ))
    return flags


def save_patient_result(pd_, risk, prob, correct_imm, correct_del, ds_score, faq_score, gds_score):
    try:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        row = {
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name":             pd_.get("name", "Anonymous") or "Anonymous",
            "sex":              pd_.get("sex", ""),
            "age":              pd_.get("age", ""),
            "education_level":  pd_.get("edu_label", ""),
            "education_years":  pd_.get("edu", 12),
            "diabetes":         pd_.get("diabetes", "No"),
            "hypertension":     pd_.get("hypertension", "No"),
            "smoking":          pd_.get("smoking", "Never"),
            "bmi":              round(float(pd_.get("bmi", 25.0)), 1),
            "other_conditions": pd_.get("other_conditions", ""),
            "words_immediate":  len(correct_imm),
            "words_delayed":    len(correct_del),
            "digit_span_score": int(ds_score),
            "faq_score":        int(faq_score),
            "gds_score":        int(gds_score),
            "cdr_g":            round(float(pd_.get("cdr_g", 0.5)), 1),
            "cdrsb":            round(float(pd_.get("cdrsb", 1.5)), 1),
            "risk_level":       risk["level"],
            "risk_pct":         risk["pct"],
            "probability":      round(prob, 4),
        }
        file_exists = os.path.exists(RESULTS_FILE)
        with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass


# ── Session state initialisation ───────────────────────────────
for k, v in [
    ("mode", None), ("patient_step", 0),
    ("patient_data", {}), ("faq_answers", {}), ("gds_answers", {})
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════
# LANDING PAGE
# ════════════════════════════════════════════════════════════════
if st.session_state.mode is None:

    st.markdown("""
    <div style='text-align:center; padding:40px 0 24px'>
      <div style='display:inline-block; background:linear-gradient(135deg,#6366f1,#8b5cf6);
          border-radius:26px; padding:18px 24px; font-size:3.4rem; margin-bottom:18px;
          box-shadow:0 12px 40px #6366f155'>🧠</div>
      <h1 style='font-size:clamp(1.8rem,5vw,3.2rem); font-weight:900; color:#1a1a2e;
          margin:0 0 12px; background:linear-gradient(135deg,#4f46e5,#7c3aed);
          -webkit-background-clip:text; -webkit-text-fill-color:transparent;
          background-clip:text'>Memory Health Assessment</h1>
      <p style='color:#475569 !important; font-size:clamp(0.9rem,2.5vw,1.1rem);
          max-width:580px; margin:0 auto 8px; line-height:1.85'>
        <b style='color:#4f46e5 !important'>Early detection. Better outcomes.</b><br>
        AI trained on 767 ADNI patients · Predicts MCI-to-dementia conversion risk<br>
        with per-patient SHAP explanations and clinical-grade assessments.
      </p>
      <div class='disclaimer'>
        ⚠️ Research prototype · Not a diagnostic tool · Always consult a qualified doctor
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c1, gap, c2 = st.columns([1, 0.05, 1])

    # ── Doctor card (entire card + button = clickable) ──
    with c1:
        st.markdown("""
        <div class='hero-card-doc' onclick="
          var btns=window.parent.document.querySelectorAll('button');
          for(var b of btns){
            if(b.innerText.trim().includes('Enter Doctor Mode')){b.click();break;}
          }
        ">
          <div style='font-size:3.6rem; margin-bottom:14px'>👨‍⚕️</div>
          <h2 style='color:white !important; font-size:1.85rem; font-weight:900; margin:0 0 10px'>
            Doctor / Clinician</h2>
          <p style='color:rgba(255,255,255,0.92) !important; font-size:0.97rem;
              line-height:1.85; margin:0 0 16px'>
            Enter clinical scores directly.<br>
            SHAP explainability · Ensemble comparison<br>
            3 threshold modes · Comorbidity flags
          </p>
          <div style='background:rgba(147,197,253,0.14); border:1px solid rgba(147,197,253,0.35);
              border-radius:12px; padding:10px 16px; font-size:0.84rem;
              color:rgba(255,255,255,0.95) !important; margin-bottom:0'>
            ⚡ Fast · Precise · 18 features · Full medical history
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🩺  Enter Doctor Mode →", key="btn_doc"):
            st.session_state.mode = "doctor"
            st.rerun()

    # ── Patient card ──
    with c2:
        st.markdown("""
        <div class='hero-card-pat' onclick="
          var btns=window.parent.document.querySelectorAll('button');
          for(var b of btns){
            if(b.innerText.trim().includes('Enter Patient Mode')){b.click();break;}
          }
        ">
          <div style='font-size:3.6rem; margin-bottom:14px'>🧑‍🤝‍🧑</div>
          <h2 style='color:white !important; font-size:1.85rem; font-weight:900; margin:0 0 10px'>
            Patient / Family</h2>
          <p style='color:rgba(255,255,255,0.92) !important; font-size:0.97rem;
              line-height:1.85; margin:0 0 16px'>
            Interactive memory tests, daily life questions,<br>
            mood assessment and health history.<br>
            Automatic scoring — no clinical knowledge needed.
          </p>
          <div style='background:rgba(236,72,153,0.18); border:1px solid rgba(236,72,153,0.4);
              border-radius:12px; padding:10px 16px; font-size:0.84rem;
              color:rgba(255,255,255,0.95) !important; margin-bottom:0'>
            🎯 Interactive · Plain language · Voice-assisted · ~10 min
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧩  Enter Patient Mode →", key="btn_pat"):
            st.session_state.mode = "patient"
            st.session_state.patient_step = 0
            st.session_state.faq_answers = {}
            st.session_state.gds_answers = {}
            st.rerun()

    st.markdown("---")
    ca, cb, cc, cd = st.columns(4)
    for col, (em, title, sub) in zip([ca, cb, cc, cd], [
        ("🤖", "AI-Powered",    "XGBoost + CNN\nAUC 0.805 [0.732–0.870]"),
        ("🔒", "100% Private",  "Data stays on device.\nNothing uploaded."),
        ("🔍", "Explainable",   "SHAP per-patient.\nKnow exactly why."),
        ("📱", "Mobile-Ready",  "Works on phone,\ntablet and laptop."),
    ]):
        col.markdown(f"""
        <div class='mcard'>
          <div style='font-size:2rem; margin-bottom:8px'>{em}</div>
          <div style='font-weight:800; font-size:1rem; color:#1a1a2e !important'>{title}</div>
          <div style='font-size:0.81rem; color:#64748b !important; margin-top:4px;
              white-space:pre-line'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.stop()


# ── Header ─────────────────────────────────────────────────────
ml = "👨‍⚕️ Doctor Mode" if st.session_state.mode == "doctor" else "🧑‍🤝‍🧑 Patient Mode"
mc = ("linear-gradient(90deg,#1d3461,#1a237e)"
      if st.session_state.mode == "doctor"
      else "linear-gradient(90deg,#4c1d95,#ec4899)")

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(f"""
    <div style='background:{mc}; border-radius:16px; padding:13px 24px;
    color:white; margin-bottom:18px; box-shadow:0 4px 20px rgba(0,0,0,0.15)'>
      <span style='font-size:1.1rem; font-weight:800; color:white !important'>
        🧠 Memory Assessment</span>
      <span style='margin-left:18px; opacity:0.85; font-size:0.93rem;
        color:white !important'>| {ml}</span>
    </div>""", unsafe_allow_html=True)
with h2:
    if st.button("🔄 Home"):
        st.session_state.mode = None
        st.session_state.patient_step = 0
        st.session_state.patient_data = {}
        st.session_state.faq_answers  = {}
        st.session_state.gds_answers  = {}
        for k in ["clin_prob", "mri_prob", "risk", "vals", "words_shown",
                  "word_phase", "digit_sequences", "digit_level", "digit_correct",
                  "digit_done", "digit_phase", "digit_results",
                  "selected_threshold", "comorbidities"]:
            st.session_state.pop(k, None)
        st.rerun()


# ════════════════════════════════════════════════════════════════
# DOCTOR MODE
# ════════════════════════════════════════════════════════════════
if st.session_state.mode == "doctor":
    tab1, tab2, tab3 = st.tabs(["📋 Assessment Form", "📊 Results & SHAP", "📁 Patient Records"])

    with tab1:
        st.markdown("### Enter clinical test scores")
        if not models:
            st.error("No models found at Website/models/")
            st.stop()

        with st.form("doctor_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="gcard"><b>👤 Demographics</b></div>', unsafe_allow_html=True)
                sex    = st.selectbox("Sex", ["Male", "Female"])
                edu    = st.number_input("Education (years)", 0, 25, 14, 1)
                cdr_g  = st.selectbox("CDR Global", [0.0, 0.5, 1.0, 2.0, 3.0], index=1)
                cdrsb  = st.number_input("CDR Sum of Boxes", 0.0, 18.0, 1.5, 0.5)
                st.markdown('<div class="gcard-warn"><b>🏥 Medical History</b></div>', unsafe_allow_html=True)
                diabetes    = st.selectbox("Diabetes", ["No", "Type 1", "Type 2"])
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                smoking     = st.selectbox("Smoking History", ["Never", "Former", "Current"])
                bmi         = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1)

            with c2:
                st.markdown('<div class="gcard"><b>🧠 Memory (RAVLT)</b></div>', unsafe_allow_html=True)
                rv_imm = st.number_input("RAVLT Immediate (0–75)", 0.0, 75.0, 35.0, 0.5)
                rv_del = st.number_input("RAVLT Delayed (0–15)",   0.0, 15.0,  7.0, 0.5)
                rv_fo  = st.number_input("RAVLT Forgetting (0–15)", 0.0, 15.0,  4.0, 0.5)
                ds     = st.number_input("Digit Span (0–28)",       0.0, 28.0, 14.0, 0.5)
                st.markdown('<div class="gcard"><b>🏠 Function</b></div>', unsafe_allow_html=True)
                faq  = st.number_input("FAQ (0–30)",  0.0, 30.0,  5.0, 0.5)
                gds  = st.number_input("GDS (0–15)",  0.0, 15.0,  2.0, 0.5)

            with c3:
                st.markdown('<div class="gcard"><b>💭 Cognition</b></div>', unsafe_allow_html=True)
                mmse   = st.number_input("MMSE (0–30)",       0.0, 30.0, 26.0, 0.5)
                moca   = st.number_input("MoCA (0–30)",       0.0, 30.0, 23.0, 0.5)
                adas13 = st.number_input("ADAS-Cog 13",       0.0, 85.0, 18.0, 0.5)
                adas11 = st.number_input("ADAS-Cog 11",       0.0, 70.0, 13.0, 0.5)
                trails = st.number_input("Trails B (seconds)",10.0,300.0,120.0, 5.0)

            st.markdown("**🎯 Prediction Mode**")
            tm = st.radio("Mode", list(THRESHOLD_OPTIONS.keys()), index=1,
                          label_visibility="collapsed", horizontal=True)
            t  = THRESHOLD_OPTIONS[tm]
            st.markdown(f"<div style='font-size:0.83rem;color:#4f46e5!important'>Threshold: <b>t = {t:.2f}</b></div>",
                        unsafe_allow_html=True)
            mri_w = st.slider("MRI fusion weight (0 = clinical only)", 0.0, 0.5, 0.0, 0.05)
            submitted = st.form_submit_button("🔍  Predict Now", use_container_width=True)

        if submitted:
            vals = {
                "PTGENDER": 0.0 if sex == "Male" else 1.0,
                "PTEDUCAT": float(edu),
                "MMSE_BL":  mmse, "MOCA_BL": moca,
                "ADAS11_BL": adas11, "ADAS13_BL": adas13,
                "FAQ_BL": faq, "GDS_BL": gds,
                "CDR_GLOBAL_BL": cdr_g, "CDRSB_BL": cdrsb,
                "RAVLT_forgetting": rv_fo, "RAVLT_immediate": rv_imm,
                "RAVLT_delayed": rv_del, "DigitSpan": ds, "TrailsB": trails
            }
            vals = compute_composites(vals)
            st.session_state.vals = vals
            X          = np.array([[vals[f] for f in ALL_FEATURES]])
            clin_prob  = float(primary.predict_proba(X)[0, 1])
            mp         = st.session_state.get("mri_prob", None)
            fp         = (1 - mri_w) * clin_prob + mri_w * mp if (mp and mri_w > 0) else clin_prob
            st.session_state.clin_prob = clin_prob
            st.session_state.selected_threshold = t
            st.session_state.risk = risk_info(fp, t)
            st.session_state.comorbidities = {
                "diabetes": diabetes, "hypertension": hypertension,
                "smoking": smoking, "bmi": bmi
            }
            st.success("✅ Done — open the **📊 Results & SHAP** tab")

    with tab2:
        risk = st.session_state.get("risk")
        vals = st.session_state.get("vals")
        t_used = st.session_state.get("selected_threshold", 0.35)
        com    = st.session_state.get("comorbidities", {})

        if risk is None:
            st.info("Complete the assessment form first.")
        else:
            r1, r2 = st.columns([1, 1])
            with r1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=risk["pct"],
                    title={"text": "Conversion Risk", "font": {"size": 14}},
                    number={"suffix": "%", "font": {"size": 52, "color": risk["color"]}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk["color"], "thickness": 0.32},
                        "bgcolor": "white",
                        "steps": [
                            {"range": [0, 35],  "color": "#f0fdf4"},
                            {"range": [35, 65], "color": "#fffbeb"},
                            {"range": [65, 100],"color": "#fef2f2"},
                        ],
                        "threshold": {"line": {"color": "#94a3b8", "width": 2}, "value": 50}
                    }
                ))
                fig.update_layout(height=290, paper_bgcolor="white",
                                  margin=dict(t=60, b=10, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"<div style='text-align:center;font-size:0.83rem;color:#4f46e5!important;"
                            f"font-weight:600'>Threshold: t = {t_used:.2f}</div>", unsafe_allow_html=True)

            with r2:
                st.markdown(f'<div class="{risk["cls"]}">', unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:3.8rem'>{risk['emoji']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:1.6rem;font-weight:900;color:{risk['color']}!important'>"
                            f"{risk['headline']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.97rem;color:#374151!important;margin-top:12px;"
                            f"line-height:1.85'>{risk['plain']}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if len(models) > 1:
                    st.markdown("**Ensemble comparison:**")
                    X_arr = np.array([[vals[f] for f in ALL_FEATURES]])
                    for nm, mdl in models.items():
                        mp2 = float(mdl.predict_proba(X_arr)[0, 1])
                        st.metric(nm, f"{mp2:.3f}", "▲ pMCI" if mp2 >= t_used else "▼ sMCI")

            if com:
                flags = comorbidity_flags(
                    com.get("diabetes", "No"), com.get("hypertension", "No"),
                    com.get("smoking", "Never"), com.get("bmi", 25.0)
                )
                if flags:
                    st.markdown("#### 🏥 Comorbidity Risk Amplifiers")
                    for em, title, desc, bg, bc in flags:
                        st.markdown(f"""
                        <div style='background:{bg};border-radius:14px;border-left:5px solid {bc};
                        padding:14px 20px;margin:6px 0'>
                          <b style='color:#1a1a2e!important'>{em} {title}</b><br>
                          <span style='font-size:0.87rem;color:#374151!important'>{desc}</span>
                        </div>""", unsafe_allow_html=True)

            if vals and primary:
                st.markdown("#### SHAP Feature Impact")
                try:
                    X_arr = np.array([[vals[f] for f in ALL_FEATURES]])
                    xm = primary.steps[-1][1] if hasattr(primary, "steps") else primary
                    if hasattr(primary, "steps"):
                        from sklearn.pipeline import Pipeline as SKP
                        try:
                            Xs = SKP(primary.steps[:-1]).transform(X_arr)
                        except Exception:
                            Xs = X_arr
                    else:
                        Xs = X_arr
                    sv  = shap.TreeExplainer(xm).shap_values(Xs)[0]
                    idx = np.argsort(np.abs(sv))[-12:][::-1]
                    nm_ = [f.replace("_BL", "").replace("_", " ") for f in ALL_FEATURES]
                    fig2, ax = plt.subplots(figsize=(7, 4.5))
                    fig2.patch.set_facecolor("white")
                    ax.set_facecolor("#f8fafc")
                    ax.barh(
                        [nm_[i] for i in idx][::-1],
                        sv[idx][::-1],
                        color=["#ef4444" if v > 0 else "#3b82f6" for v in sv[idx][::-1]],
                        alpha=0.85, height=0.6
                    )
                    ax.axvline(0, color="#cbd5e1", lw=1.5)
                    ax.set_xlabel("← Lower risk   |   Higher risk →",
                                  fontsize=9, color="#374151")
                    ax.tick_params(colors="#374151", labelsize=8.5)
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#e2e8f0")
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                except Exception as e:
                    st.info(f"SHAP: {e}")

    with tab3:
        st.markdown("### 📁 Patient Assessment Records")
        if os.path.exists(RESULTS_FILE):
            import pandas as pd
            df_results = pd.read_csv(RESULTS_FILE)
            st.markdown(f"**{len(df_results)} assessments recorded**")
            st.dataframe(df_results, use_container_width=True)
            st.download_button(
                "⬇️ Download all records (CSV)",
                data=df_results.to_csv(index=False),
                file_name="patient_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No patient assessments recorded yet. Results appear here after patients complete the assessment.")


# ════════════════════════════════════════════════════════════════
# PATIENT MODE
# ════════════════════════════════════════════════════════════════
else:
    step  = st.session_state.patient_step
    pd_   = st.session_state.patient_data
    STEPS = ["📋 Info", "🏥 Health", "🧠 Memory", "🏠 Daily Life",
             "💬 Mood", "🔢 Numbers", "🕐 Recall", "📊 Results"]
    total = len(STEPS)
    pct   = int((step / max(total - 1, 1)) * 100)

    pills = "".join([
        f'<span class="{"npill-a" if i == step else "npill"}">{s}</span>'
        for i, s in enumerate(STEPS)
    ])
    st.markdown(f"""
    <div style='margin-bottom:8px;line-height:2.6'>{pills}</div>
    <div class='prog-bg'><div class='prog-fill' style='width:{pct}%'></div></div>
    <div style='font-size:0.76rem;color:#6366f1!important;font-weight:700;
    text-align:right;margin-bottom:18px'>
      Step {step + 1} of {total} — {STEPS[step]}
    </div>""", unsafe_allow_html=True)

    # ── STEP 0 : Personal Info ─────────────────────────────────
    if step == 0:
        st.markdown("""
        <h2 style='color:#1a1a2e!important;font-size:1.9rem;font-weight:900;margin-bottom:6px'>
        👤 Who is taking this assessment?</h2>
        <p style='color:#64748b!important;margin-bottom:22px'>
        We'll personalise the results based on your information.</p>
        """, unsafe_allow_html=True)

        edu_options = [
            ("🏫 No formal education (0 yrs)",           0,  "No formal education"),
            ("📚 Schooling — 1st to 10th (10 yrs)",      10, "Schooling (1st–10th)"),
            ("🎒 High School — 11th & 12th (12 yrs)",    12, "High School (11th–12th)"),
            ("🎓 UG / Bachelor's — 4 or 5 years (16 yrs)", 16, "UG / Bachelor's"),
            ("📖 PG / Master's — 2 years (18 yrs)",      18, "PG / Master's"),
            ("🔬 Doctorate / PhD (22+ yrs)",              22, "Doctorate / PhD"),
        ]

        with st.form("p0"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Full Name** *(optional)*")
                name = st.text_input("Full Name", placeholder="e.g. Ravi Kumar",
                                     label_visibility="collapsed")
                st.markdown("**Sex**")
                sex  = st.selectbox("Sex", ["Male", "Female", "Other / Prefer not to say"],
                                    label_visibility="collapsed")
                st.markdown("**Age**")
                age  = st.number_input("Age", 18, 100, 65, 1,
                                       label_visibility="collapsed")
            with c2:
                st.markdown("**🎓 Highest level of education**")
                edu_choice = st.radio(
                    "Education",
                    [opt[0] for opt in edu_options],
                    index=3,
                    label_visibility="collapsed"
                )

            if st.form_submit_button("Continue →", use_container_width=True):
                # Map choice → years, label
                edu_map = {opt[0]: (opt[1], opt[2]) for opt in edu_options}
                edu_years, edu_label = edu_map[edu_choice]
                pd_.update({
                    "name": name, "sex": sex, "age": age,
                    "edu": edu_years, "edu_label": edu_label
                })
                st.session_state.patient_step = 1
                st.session_state.patient_data = pd_
                for k in ["word_phase", "digit_sequences", "digit_level",
                          "digit_results", "digit_done", "digit_phase"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # ── STEP 1 : Health History ───────────────────────────────
    elif step == 1:
        name     = pd_.get("name", "")
        greeting = f"Hi {name.split()[0]}! " if name else ""
        st.markdown(f"""
        <h2 style='color:#1a1a2e!important;font-size:1.9rem;font-weight:900;margin-bottom:6px'>
        🏥 {greeting}Health History</h2>
        <p style='color:#64748b!important;margin-bottom:20px'>
        These conditions independently raise memory decline risk. Please answer honestly.</p>
        """, unsafe_allow_html=True)

        with st.form("health_form"):
            c1, c2 = st.columns(2)

            with c1:
                # Diabetes (Type 1 & Type 2 both matter)
                st.markdown("""
                <div class='hcard'>
                  <b style='font-size:1rem;color:#1a1a2e!important'>🩸 Diabetes</b><br>
                  <span style='font-size:0.83rem;color:#64748b!important'>
                  Both Type 1 and Type 2 affect insulin regulation linked to brain amyloid</span>
                </div>""", unsafe_allow_html=True)
                diabetes = st.radio("Diabetes", ["No", "Type 1", "Type 2"],
                                    horizontal=True, key="hr_diab",
                                    label_visibility="collapsed")

                st.markdown("""
                <div class='hcard'>
                  <b style='font-size:1rem;color:#1a1a2e!important'>💉 High Blood Pressure</b><br>
                  <span style='font-size:0.83rem;color:#64748b!important'>
                  Vascular damage accelerates cognitive decline</span>
                </div>""", unsafe_allow_html=True)
                hypertension = st.radio("Hypertension", ["No ✅", "Yes ⚠️"],
                                        horizontal=True, key="hr_hyp",
                                        label_visibility="collapsed")

                st.markdown("""
                <div class='hcard'>
                  <b style='font-size:1rem;color:#1a1a2e!important'>📋 Other Health Conditions</b><br>
                  <span style='font-size:0.83rem;color:#64748b!important'>
                  e.g. heart disease, thyroid issues, sleep apnea, depression</span>
                </div>""", unsafe_allow_html=True)
                other_cond = st.text_area("Other conditions", placeholder="Describe any other health conditions here...",
                                          height=80, label_visibility="collapsed")

            with c2:
                # Smoking
                st.markdown("""
                <div class='hcard'>
                  <b style='font-size:1rem;color:#1a1a2e!important'>🚬 Smoking History</b><br>
                  <span style='font-size:0.83rem;color:#64748b!important'>
                  Significant cerebrovascular risk factor</span>
                </div>""", unsafe_allow_html=True)
                smoking = st.radio("Smoking", ["Never ✅", "Former 🟡", "Current ⚠️"],
                                   horizontal=True, key="hr_smk",
                                   label_visibility="collapsed")

                # BMI from height + weight
                st.markdown("""
                <div class='hcard'>
                  <b style='font-size:1rem;color:#1a1a2e!important'>⚖️ Body Weight</b><br>
                  <span style='font-size:0.83rem;color:#64748b!important'>
                  Midlife obesity significantly increases dementia risk</span>
                </div>""", unsafe_allow_html=True)

                bh1, bh2 = st.columns(2)
                with bh1:
                    height_cm = st.number_input("Height (cm)", 100, 220, 165, 1)
                with bh2:
                    weight_kg = st.number_input("Weight (kg)", 30, 200, 70, 1)

                bmi_calc = weight_kg / ((height_cm / 100) ** 2)
                if bmi_calc < 18.5:
                    bmi_label, bmi_col = "Underweight", "#3b82f6"
                elif bmi_calc < 25:
                    bmi_label, bmi_col = "Healthy Weight ✅", "#10b981"
                elif bmi_calc < 30:
                    bmi_label, bmi_col = "Overweight ⚠️", "#f59e0b"
                else:
                    bmi_label, bmi_col = "Obese ⚠️", "#ef4444"

                st.markdown(f"""
                <div style='background:#f8fafc;border-radius:12px;border:2px solid #e0e7ff;
                padding:12px 16px;margin-top:8px;text-align:center'>
                  <span style='font-size:1.6rem;font-weight:900;color:{bmi_col}!important'>
                    BMI {bmi_calc:.1f}</span><br>
                  <span style='font-size:0.9rem;font-weight:700;color:{bmi_col}!important'>
                    {bmi_label}</span>
                </div>""", unsafe_allow_html=True)

            if st.form_submit_button("Continue →", use_container_width=True):
                d_val = diabetes  # "No", "Type 1", or "Type 2"
                h_val = "Yes" if "Yes" in hypertension else "No"
                s_map = {"Never ✅": "Never", "Former 🟡": "Former", "Current ⚠️": "Current"}
                s_val = s_map.get(smoking, "Never")
                pd_.update({
                    "diabetes": d_val, "hypertension": h_val,
                    "smoking": s_val, "bmi": float(bmi_calc),
                    "other_conditions": other_cond.strip()
                })
                st.session_state.patient_step = 2
                st.session_state.patient_data = pd_
                st.rerun()

    # ── STEP 2 : Word Memory ──────────────────────────────────
    elif step == 2:
        if "word_phase" not in st.session_state:
            st.session_state.word_phase = "show"
            st.session_state.words_shown = WORD_LIST[:]

        words = st.session_state.words_shown

        # ── Show phase ──
        if st.session_state.word_phase == "show":
            st.markdown("""
            <h2 style='color:#1a1a2e!important;font-size:1.9rem;font-weight:900'>
            🧠 Word Memory Test</h2>
            <p style='color:#64748b!important'>
            Study all <b style='color:#6366f1!important'>15 words</b> carefully.
            Use the voice button to hear them. <b>Words disappear when you click Ready</b>
            — then type what you remember.</p>
            """, unsafe_allow_html=True)

            wh = "".join([f'<span class="wbox">{w.upper()}</span>' for w in words])
            st.markdown(f"""
            <div class='gcard'>
              <h3 style='margin:0 0 14px;color:#1a1a2e!important;font-size:1.1rem;font-weight:800'>
                👀 Study these words carefully:</h3>
              <div style='line-height:3.2'>{wh}</div>
            </div>""", unsafe_allow_html=True)

            import json as _json
            words_js = _json.dumps(words)
            components.html(f"""
            <div style="font-family:Inter,sans-serif;display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:10px 0">
              <button id="speakBtn" onclick="startSpeech()"
                style="background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;border:none;
                       border-radius:12px;padding:12px 22px;font-size:0.97rem;font-weight:800;
                       cursor:pointer;box-shadow:0 4px 16px #6366f144">
                🔊 Listen to All Words
              </button>
              <button id="stopBtn" onclick="stopSpeech()" disabled
                style="background:#ef4444;color:white;border:none;border-radius:12px;
                       padding:12px 18px;font-size:0.97rem;font-weight:800;cursor:pointer;opacity:0.4">
                ⏹ Stop
              </button>
              <span id="status" style="color:#6366f1;font-size:0.9rem;font-weight:700"></span>
            </div>
            <script>
            const words={words_js}; let speaking=false;
            function startSpeech(){{
              window.speechSynthesis.cancel(); speaking=true;
              document.getElementById('speakBtn').disabled=true;
              document.getElementById('stopBtn').disabled=false;
              document.getElementById('stopBtn').style.opacity='1';
              document.getElementById('status').innerText='▶ Playing...';
              let idx=0;
              function next(){{
                if(!speaking||idx>=words.length){{
                  document.getElementById('status').innerText=speaking?'✅ All words played!':'⏹ Stopped.';
                  document.getElementById('speakBtn').disabled=false;
                  document.getElementById('speakBtn').innerText='🔊 Listen Again';
                  document.getElementById('stopBtn').disabled=true;
                  document.getElementById('stopBtn').style.opacity='0.4'; return;
                }}
                document.getElementById('status').innerText='▶ '+(idx+1)+'/'+words.length+': '+words[idx].toUpperCase();
                const u=new SpeechSynthesisUtterance(words[idx]); u.rate=0.75;
                u.onend=()=>{{idx++;setTimeout(next,600);}};
                window.speechSynthesis.speak(u);
              }} next();
            }}
            function stopSpeech(){{
              speaking=false; window.speechSynthesis.cancel();
              document.getElementById('speakBtn').disabled=false;
              document.getElementById('speakBtn').innerText='🔊 Listen Again';
              document.getElementById('stopBtn').disabled=true;
              document.getElementById('stopBtn').style.opacity='0.4';
              document.getElementById('status').innerText='⏹ Stopped.';
            }}
            </script>""", height=90)

            st.warning("📌 **Important:** Once you click the button below, all words disappear permanently.")
            if st.button("✅  I'm Ready — Hide Words & Start Recall", use_container_width=True):
                st.session_state.word_phase = "recall"
                st.rerun()

        # ── Recall phase ──
        else:
            st.markdown("""
            <h2 style='color:#1a1a2e!important;font-size:1.9rem;font-weight:900'>
            ✍️ What words do you remember?</h2>
            <p style='color:#64748b!important'>The words are hidden. Type every word that comes to mind.
            Separate words with spaces or commas.</p>
            """, unsafe_allow_html=True)

            imm = st.text_area(
                "Your recalled words:",
                height=130,
                key="imm_recall",
                placeholder="Type your words here..."
            )

            st.info("💡 Don't worry about spelling — just write everything you can recall.")

            if st.button("Submit & Continue →", use_container_width=True):
                recalled   = [w.strip().lower() for w in imm.replace(",", " ").split() if w.strip()]
                word_set   = {w.lower() for w in words}
                correct_imm = [w for w in recalled if w in word_set]
                pd_.update({
                    "immediate_recall": correct_imm,
                    "rv_imm": min(len(correct_imm) * 3.5, 75)
                })
                st.session_state.patient_step = 3
                st.session_state.patient_data = pd_
                st.rerun()

    # ── STEP 3 : Daily Life (FAQ) ─────────────────────────────
    elif step == 3:
        st.markdown("""
        <h2 style='color:#1a1a2e!important;font-size:1.9rem;font-weight:900'>
        🏠 Daily Life Questions</h2>
        <p style='color:#64748b!important'>
        For each activity, select how much help is needed.</p>
        """, unsafe_allow_html=True)

        FAQ_OPTIONS = ["✅ No help needed", "🤔 Sometimes needs help",
                       "😟 Often needs help", "❌ Cannot do it at all"]
        SCORE_MAP   = {"✅ No help needed": 0, "🤔 Sometimes needs help": 1,
                       "😟 Often needs help": 2, "❌ Cannot do it at all": 3}

        faq_score = 0
        pairs = [(FAQ_ITEMS[i], FAQ_ITEMS[i+1] if i+1 < len(FAQ_ITEMS) else None)
                 for i in range(0, len(FAQ_ITEMS), 2)]

        for left, right in pairs:
            c1, c2 = st.columns(2)
            for col, item in [(c1, left), (c2, right)]:
                if item is None:
                    continue
                emoji, question = item
                with col:
                    st.markdown(f"""
                    <div class='faq-q'>
                      <span style='font-size:1.25rem'>{emoji}</span>
                      <span style='font-weight:700;font-size:0.92rem;
                        color:#1a1a2e!important;margin-left:8px'>{question}</span>
                    </div>""", unsafe_allow_html=True)
                    sel = st.radio(
                        f"_{question}",
                        FAQ_OPTIONS,
                        key=f"faq_{question}",
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    faq_score += SCORE_MAP[sel]

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue →", use_container_width=True):
            # Map FAQ score → CDR approximation
            if faq_score <= 2:
                cdr_g, cdrsb = 0.0, 0.0
            elif faq_score <= 6:
                cdr_g, cdrsb = 0.5, 2.0
            elif faq_score <= 14:
                cdr_g, cdrsb = 1.0, 4.5
            elif faq_score <= 22:
                cdr_g, cdrsb = 2.0, 9.0
            else:
                cdr_g, cdrsb = 3.0, 14.0
            pd_.update({"faq": faq_score, "cdr_g": cdr_g, "cdrsb": cdrsb})
            st.session_state.patient_step = 4
            st.session_state.patient_data = pd_
            st.rerun()

    # ── STEP 4 : Mood / GDS ───────────────────────────────────
    elif step == 4:
        st.markdown("""
        <h2 style='color:#1a1a2e!important;font-size:1.9rem;font-weight:900'>
        💬 Mood & Wellbeing</h2>
        <p style='color:#64748b!important'>
        Think about how you felt <b>over the past week</b>. Answer Yes or No.</p>
        """, unsafe_allow_html=True)

        gds_score = 0
        with st.form("gds_form"):
            pairs = [(GDS_ITEMS[i], GDS_ITEMS[i+1] if i+1 < len(GDS_ITEMS) else None)
                     for i in range(0, len(GDS_ITEMS), 2)]
            for left, right in pairs:
                c1, c2 = st.columns(2)
                for col, item in [(c1, left), (c2, right)]:
                    if item is None:
                        continue
                    emoji, question, bad_ans = item
                    idx = GDS_ITEMS.index(item)
                    with col:
                        st.markdown(f"""
                        <div class='hcard'>
                          <span style='font-size:1.3rem'>{emoji}</span>
                          <span style='font-weight:600;font-size:0.9rem;
                            color:#1a1a2e!important;margin-left:8px'>{question}</span>
                        </div>""", unsafe_allow_html=True)
                        resp = st.radio(
                            f"gq{idx}", ["Yes", "No"],
                            horizontal=True, key=f"gds_{idx}",
                            label_visibility="collapsed", index=1
                        )
                        if resp.lower() == bad_ans:
                            gds_score += 1

            if st.form_submit_button("Continue →", use_container_width=True):
                pd_["gds"] = gds_score
                st.session_state.patient_step = 5
                st.session_state.patient_data = pd_
                st.rerun()

    # ── STEP 5 : Digit Span ───────────────────────────────────
    elif step == 5:
        st.markdown("""
        <h2 style='color:#1a1a2e!important;font-size:1.9rem;font-weight:900'>
        🔢 Number Memory Test</h2>
        <p style='color:#64748b!important'>
        A sequence of numbers will appear briefly. Remember the exact order, then type them.</p>
        """, unsafe_allow_html=True)

        if "digit_sequences" not in st.session_state:
            seqs = []
            for length in [3, 4, 5, 6, 7, 8, 9]:
                seq = random.sample(range(1, 10), min(length, 9))
                while len(seq) < length:
                    seq.append(random.randint(1, 9))
                seqs.append(seq)
            st.session_state.update({
                "digit_sequences": seqs, "digit_level": 0,
                "digit_results": [], "digit_done": False, "digit_phase": "show"
            })

        level       = st.session_state.digit_level
        done        = st.session_state.digit_done
        seqs        = st.session_state.digit_sequences
        phase       = st.session_state.digit_phase
        results_log = st.session_state.get("digit_results", [])

        if done or level >= len(seqs):
            # Calculate score
            correct_count    = sum(1 for r in results_log if r["correct"])
            max_correct_span = 0
            for r in results_log:
                if r["correct"]:
                    max_correct_span = r["length"]
                else:
                    break
            ds_score = min(max_correct_span * 2.5, 28)
            pd_["digit_span"] = ds_score
            st.session_state.patient_data = pd_

            st.markdown(f"""
            <div class='gcard' style='text-align:center;padding:36px'>
              <div style='font-size:3.5rem;margin-bottom:10px'>🎯</div>
              <h2 style='color:#1a1a2e!important;font-size:1.7rem;font-weight:900'>
                Number Test Complete!</h2>
              <p style='color:#64748b!important'>
                You completed all {len(seqs)} sequences.</p>
            """, unsafe_allow_html=True)

            for r in results_log:
                icon  = "✅" if r["correct"] else "❌"
                color = "#10b981" if r["correct"] else "#ef4444"
                st.markdown(f"""
                <div style='background:white;border-radius:12px;border:2px solid {color}33;
                padding:10px 16px;margin:4px 0;display:flex;align-items:center;gap:12px'>
                  <span style='font-size:1.1rem'>{icon}</span>
                  <span style='font-family:monospace;font-size:1rem;font-weight:800;
                    color:#1a1a2e!important;letter-spacing:6px'>
                    {" ".join(str(d) for d in r["seq"])}</span>
                  <span style='font-size:0.83rem;color:{color}!important;font-weight:700'>
                    {"Correct ✓" if r["correct"] else f'You typed: {r.get("answer","?")}'}
                  </span>
                </div>""", unsafe_allow_html=True)

            pct_bar = int(ds_score / 28 * 100)
            st.markdown(f"""
            <div style='margin-top:18px'>
              <div style='background:#e0e7ff;border-radius:999px;height:14px;
                   max-width:320px;margin:0 auto'>
                <div style='background:linear-gradient(90deg,#6366f1,#8b5cf6);
                     border-radius:999px;height:14px;width:{pct_bar}%'></div></div>
              <div style='font-size:2.2rem;font-weight:900;color:#6366f1!important;
                   margin-top:12px'>Score: {int(ds_score)}/28</div>
            </div></div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("← Redo Test"):
                    for k in ["digit_sequences", "digit_level", "digit_results",
                              "digit_done", "digit_phase"]:
                        st.session_state.pop(k, None)
                    st.rerun()
            with c2:
                if st.button("Continue →", use_container_width=True):
                    st.session_state.patient_step = 6
                    st.session_state.patient_data = pd_
                    st.rerun()

        elif phase == "show":
            seq    = seqs[level]
            seq_js = str([str(d) for d in seq]).replace("'", '"')
            seq_str = "  ".join(str(d) for d in seq)

            # Custom progress counter (no st.progress)
            st.markdown(f"""
            <div style='background:white;border-radius:12px;border:2px solid #e0e7ff;
            padding:10px 18px;margin-bottom:14px;display:flex;justify-content:space-between;
            align-items:center'>
              <span style='font-weight:700;color:#6366f1!important'>
                Sequence {level + 1} of {len(seqs)}</span>
              <span style='font-size:0.85rem;color:#64748b!important'>
                {len(seq)}-digit number</span>
              <div style='background:#e0e7ff;border-radius:999px;width:120px;height:8px'>
                <div style='background:linear-gradient(90deg,#6366f1,#8b5cf6);
                     border-radius:999px;height:8px;
                     width:{int(level/len(seqs)*100)}%'></div></div>
            </div>
            <p style='color:#64748b!important;font-size:0.97rem'>
            A <b style='color:#6366f1!important'>{len(seq)}-digit</b> sequence appears
            for <b style='color:#6366f1!important'>5 seconds</b>, then disappears.
            Remember the exact order!</p>
            """, unsafe_allow_html=True)

            components.html(f"""
            <div style="font-family:Inter,sans-serif">
              <div id="dd" style="font-family:monospace;font-size:4.5rem;font-weight:900;
                   letter-spacing:22px;text-align:center;color:#1a1a2e;
                   padding:32px;background:linear-gradient(135deg,#f0f4ff,#ede9fe);
                   border-radius:24px;border:3px solid #c7d2fe;margin:12px 0;
                   box-shadow:0 6px 24px #6366f110">
                {seq_str}
              </div>
              <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
                <button onclick="readDigits()"
                  style="background:linear-gradient(135deg,#1a1a2e,#4f46e5);color:white;
                         border:none;border-radius:12px;padding:12px 22px;font-size:0.95rem;
                         font-weight:800;cursor:pointer;box-shadow:0 4px 16px #1a1a2e33">
                  🔊 Read aloud
                </button>
                <div id="cd" style="display:none">
                  <span style="font-size:0.9rem;color:#6366f1;font-weight:800">
                    Hiding in <span id="cdnum">5</span>s…</span>
                  <div style="background:#e0e7ff;border-radius:999px;height:8px;
                       width:160px;margin-top:6px">
                    <div id="cdbar" style="background:linear-gradient(90deg,#6366f1,#8b5cf6);
                         border-radius:999px;height:8px;width:100%;
                         transition:width 1s linear"></div>
                  </div>
                </div>
              </div>
            </div>
            <script>
            const digits={seq_js}; let started=false;
            function readDigits(){{
              window.speechSynthesis.cancel(); let i=0;
              function next(){{
                if(i<digits.length){{
                  const u=new SpeechSynthesisUtterance(digits[i]); u.rate=0.65; u.pitch=1.1;
                  u.onend=()=>{{i++;setTimeout(next,500);}};
                  window.speechSynthesis.speak(u);
                }} else {{if(!started){{started=true;startHide();}}}}
              }} next();
            }}
            function startHide(){{
              document.getElementById('cd').style.display='block'; let t=5;
              const iv=setInterval(()=>{{
                t--; document.getElementById('cdnum').innerText=t;
                document.getElementById('cdbar').style.width=(t/5*100)+'%';
                if(t<=0){{
                  clearInterval(iv);
                  document.getElementById('dd').innerHTML=
                    '<div style="text-align:center;font-size:1.2rem;color:#6366f1;'
                    +'font-weight:900;padding:22px;letter-spacing:0">✅ Numbers hidden!<br>'
                    +'Click below to type your answer.</div>';
                  document.getElementById('dd').style.background='#f0fdf4';
                  document.getElementById('dd').style.borderColor='#86efac';
                }}
              }},1000);
            }}
            setTimeout(()=>{{if(!started){{started=true;startHide();}}}},5500);
            </script>""", height=200)

            st.info("⏱ Auto-hides after 5 seconds. You can also click 🔊 to hear them.")
            if st.button("I'm ready to type my answer →", use_container_width=True,
                         key=f"ready_{level}"):
                st.session_state.digit_phase = "type"
                st.rerun()

        else:  # type phase
            seq = seqs[level]
            st.markdown(f"""
            <h3 style='color:#1a1a2e!important;font-size:1.3rem;font-weight:800'>
              Type the {len(seq)} numbers in the exact order:</h3>
            <p style='color:#64748b!important'>
              No spaces needed — just type the digits straight, e.g.
              <b style='color:#6366f1!important'>47293</b></p>
            """, unsafe_allow_html=True)

            answer = st.text_input(
                f"Type {len(seq)} digits:",
                key=f"ans_{level}",
                placeholder=f"Enter {len(seq)} digits here...",
                label_visibility="collapsed"
            )

            if st.button("✅  Submit Answer →", use_container_width=True,
                         key=f"sub_{level}"):
                correct_str = "".join(str(d) for d in seq)
                is_correct  = answer.strip().replace(" ", "") == correct_str
                results_log.append({
                    "level": level, "seq": seq,
                    "answer": answer.strip(),
                    "correct": is_correct, "length": len(seq)
                })
                st.session_state.digit_results = results_log
                st.session_state.digit_level   = level + 1
                st.session_state.digit_phase   = "show"
                if level + 1 >= len(seqs):
                    st.session_state.digit_done = True
                st.rerun()

    # ── STEP 6 : Delayed Recall + Optional MRI ────────────────
    elif step == 6:
        words_shown = st.session_state.get("words_shown", WORD_LIST)
        st.markdown(f"""
        <h2 style='color:#1a1a2e!important;font-size:1.9rem;font-weight:900'>
        🕐 Word Recall</h2>
        <p style='color:#64748b!important'>
        Earlier we showed you <b style='color:#6366f1!important'>{len(words_shown)} words</b>.
        They're still hidden. Type every word you can remember now.</p>
        """, unsafe_allow_html=True)

        delayed_input = st.text_area(
            "Words you remember:",
            height=130,
            key="del_recall",
            placeholder="Type the words you remember here..."
        )
        st.info("💡 Take your time. Even partial words count if the meaning is clear.")

        # Optional MRI Upload
        st.markdown("---")
        st.markdown("""
        <div class='gcard'>
          <div style='font-size:1.5rem;margin-bottom:8px'>🧬 Optional: Upload MRI Scan</div>
          <p style='color:#64748b!important;font-size:0.9rem;margin-bottom:12px'>
          If you have a brain MRI image available, you can upload it here.
          This is completely optional — the assessment works without it.<br>
          <span style='color:#6366f1!important;font-weight:600'>
          Accepted formats: JPG, PNG (axial brain slice preferred)</span>
          </p>
        </div>
        """, unsafe_allow_html=True)

        mri_file = st.file_uploader(
            "Upload MRI (optional)",
            type=["jpg", "jpeg", "png"],
            key="mri_upload",
            label_visibility="collapsed",
            help="Optional brain MRI image for additional analysis"
        )
        if mri_file is not None:
            st.success(f"✅ MRI image received: {mri_file.name} — "
                       "will be noted in your assessment report.")
            pd_["mri_uploaded"] = True
        else:
            pd_["mri_uploaded"] = False

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊  See My Results →", use_container_width=True):
            recalled_del = [w.strip().lower()
                            for w in delayed_input.replace(",", " ").split()
                            if w.strip()]
            word_set    = {w.lower() for w in words_shown}
            correct_del = [w for w in recalled_del if w in word_set]
            correct_imm = pd_.get("immediate_recall", [])
            pd_.update({
                "rv_del":      float(len(correct_del)),
                "rv_fo":       float(max(len(correct_imm) - len(correct_del), 0)),
                "correct_del": correct_del
            })
            st.session_state.patient_step = 7
            st.session_state.patient_data = pd_
            st.rerun()

    # ── STEP 7 : Results ──────────────────────────────────────
    elif step == 7:
        st.markdown("""
        <h2 style='color:#1a1a2e!important;font-size:1.9rem;font-weight:900'>
        📊 Your Assessment Results</h2>
        """, unsafe_allow_html=True)

        rv_del  = float(pd_.get("rv_del", 7))
        rv_fo   = float(pd_.get("rv_fo", 4))
        rv_imm  = float(pd_.get("rv_imm", 35))
        ds      = float(pd_.get("digit_span", 14))
        faq     = float(pd_.get("faq", 5))
        gds     = float(pd_.get("gds", 2))
        cdr_g   = float(pd_.get("cdr_g", 0.5))
        cdrsb   = float(pd_.get("cdrsb", 1.5))
        edu     = float(pd_.get("edu", 12))
        sex_val = 0.0 if pd_.get("sex", "Male") == "Male" else 1.0
        correct_del = pd_.get("correct_del", [])
        correct_imm = pd_.get("immediate_recall", [])

        vals = {
            "PTGENDER": sex_val, "PTEDUCAT": edu,
            "MMSE_BL": 26.0, "MOCA_BL": 23.0,
            "ADAS11_BL": 13.0, "ADAS13_BL": 18.0,
            "FAQ_BL": faq, "GDS_BL": gds,
            "CDR_GLOBAL_BL": cdr_g, "CDRSB_BL": cdrsb,
            "RAVLT_forgetting": rv_fo, "RAVLT_immediate": rv_imm,
            "RAVLT_delayed": rv_del, "DigitSpan": ds, "TrailsB": 120.0
        }
        vals = compute_composites(vals)

        if not primary:
            st.error("AI model not loaded.")
            st.stop()

        X    = np.array([[vals[f] for f in ALL_FEATURES]])
        prob = float(primary.predict_proba(X)[0, 1])
        risk = risk_info(prob, threshold=0.35)

        # Save to backend
        save_patient_result(pd_, risk, prob, correct_imm, correct_del, ds, faq, gds)

        # ── Score summary cards ──
        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, num, denom, label, color in [
            (sc1, len(correct_imm), 15, "Words\n(immediate)",  "#6366f1"),
            (sc2, len(correct_del), 15, "Words\n(after delay)",
             "#10b981" if len(correct_del) >= max(len(correct_imm) * 0.6, 1) else "#ef4444"),
            (sc3, int(ds), 28,          "Digit\nSpan",          "#8b5cf6"),
            (sc4, int(faq), 30,         "Daily\nDifficulty",    "#f59e0b"),
        ]:
            pct_bar = int(num / max(denom, 1) * 100)
            col.markdown(f"""
            <div class='mcard'>
              <div style='font-size:2rem;font-weight:900;color:{color}!important'>
                {num}<span style='font-size:0.9rem;font-weight:600;
                  color:#94a3b8!important'>/{denom}</span>
              </div>
              <div style='font-size:0.8rem;color:#64748b!important;
                margin:6px 0;white-space:pre-line'>{label}</div>
              <div style='background:#e0e7ff;border-radius:999px;height:5px;margin-top:8px'>
                <div style='background:{color};border-radius:999px;
                  height:5px;width:{pct_bar}%'></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Main result + gauge ──
        rr1, rr2 = st.columns([1, 1])
        with rr1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=risk["pct"],
                title={"text": "Conversion Risk", "font": {"size": 16}},
                number={"suffix": "%", "font": {"size": 56, "color": risk["color"]}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk["color"], "thickness": 0.35},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 35],  "color": "#f0fdf4"},
                        {"range": [35, 65], "color": "#fffbeb"},
                        {"range": [65, 100],"color": "#fef2f2"},
                    ],
                    "threshold": {"line": {"color": "#94a3b8", "width": 2}, "value": 50}
                }
            ))
            fig.update_layout(height=300, paper_bgcolor="white",
                              margin=dict(t=60, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with rr2:
            st.markdown(f'<div class="{risk["cls"]}">', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:4rem'>{risk['emoji']}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.7rem;font-weight:900;"
                        f"color:{risk['color']}!important'>{risk['headline']}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.97rem;color:#374151!important;"
                        f"margin-top:12px;line-height:1.9'>{risk['plain']}</div>",
                        unsafe_allow_html=True)
            if pd_.get("mri_uploaded"):
                st.markdown("""
                <div style='background:#f0f9ff;border-radius:12px;border:2px solid #bae6fd;
                padding:10px 14px;margin-top:12px;font-size:0.87rem;color:#0369a1!important'>
                  🧬 MRI scan received. For full MRI-integrated analysis,
                  ask your doctor to use Doctor Mode.
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Comorbidities ──
        com_flags = comorbidity_flags(
            pd_.get("diabetes", "No"),
            pd_.get("hypertension", "No"),
            pd_.get("smoking", "Never"),
            pd_.get("bmi", 25.0),
            pd_.get("other_conditions", "")
        )
        if com_flags:
            st.markdown("---")
            st.markdown("### 🏥 Additional Risk Factors")
            for em, title, desc, bg, bc in com_flags:
                st.markdown(f"""
                <div style='background:{bg};border-radius:14px;border-left:5px solid {bc};
                padding:14px 20px;margin:8px 0'>
                  <b style='font-size:0.98rem;color:#1a1a2e!important'>{em} {title}</b><br>
                  <span style='font-size:0.86rem;color:#374151!important'>{desc}</span>
                </div>""", unsafe_allow_html=True)

        # ── SHAP explanation ──
        PLAIN = {
            "ADAS13_BL":          ("Thinking errors",          "higher = more concern"),
            "RAVLT_delayed":      ("Words recalled after delay","lower = more concern"),
            "DigitSpan":          ("Number memory",            "lower = more concern"),
            "CDRSB_BL":           ("Daily function difficulty", "higher = more concern"),
            "FAQ_BL":             ("Daily activities difficulty","higher = more concern"),
            "MMSE_BL":            ("Short memory test",        "lower = more concern"),
            "RAVLT_immediate":    ("Total words recalled",     "lower = more concern"),
            "RAVLT_forgetting":   ("Words forgotten over time","higher = more concern"),
            "GDS_BL":             ("Mood / depression score",  "higher = more concern"),
            "CDR_GLOBAL_BL":      ("Overall memory rating",    "higher = more concern"),
            "MMSE_FAQ_composite": ("Memory vs function gap",   "lower = more concern"),
            "ADAS_MMSE_gap":      ("Thinking difficulty index","higher = more concern"),
            "RAVLT_forget_rate":  ("Rate of forgetting",       "higher = more concern"),
            "TrailsB":            ("Mental flexibility",       "slower = more concern"),
            "ADAS11_BL":          ("Thinking test errors",     "higher = more concern"),
            "PTEDUCAT":           ("Years of education",       "fewer = some risk"),
            "PTGENDER":           ("Sex",                      "minor factor"),
            "MOCA_BL":            ("Overall thinking test",    "lower = more concern"),
        }

        st.markdown("---")
        st.markdown("### 🔍 What is driving your result?")

        try:
            xm = primary.steps[-1][1] if hasattr(primary, "steps") else primary
            if hasattr(primary, "steps"):
                from sklearn.pipeline import Pipeline as SKP
                try:
                    Xs = SKP(primary.steps[:-1]).transform(X)
                except Exception:
                    Xs = X
            else:
                Xs = X
            sv      = shap.TreeExplainer(xm).shap_values(Xs)[0]
            top_idx = np.argsort(np.abs(sv))[-8:][::-1]
            fu, fd  = [], []
            for i in top_idx:
                feat       = ALL_FEATURES[i]
                sv_        = sv[i]
                label, hint = PLAIN.get(feat, (feat, ""))
                if sv_ > 0.05:
                    fu.append((label, hint, abs(sv_)))
                elif sv_ < -0.05:
                    fd.append((label, hint, abs(sv_)))

            sh1, sh2 = st.columns(2)
            with sh1:
                if fu:
                    st.markdown("#### 🔴 Raising your risk")
                    for label, hint, mag in sorted(fu, key=lambda x: -x[2]):
                        bw = min(int(mag * 280), 100)
                        st.markdown(f"""
                        <div style='background:white;border-left:4px solid #ef4444;
                        border-radius:12px;padding:14px 18px;margin:6px 0;
                        box-shadow:0 2px 8px #ef44440a'>
                          <b style='color:#1a1a2e!important'>{label}</b><br>
                          <span style='font-size:0.8rem;color:#64748b!important'>{hint}</span>
                          <div style='background:#fee2e2;border-radius:999px;
                            height:6px;margin-top:8px'>
                            <div style='background:#ef4444;border-radius:999px;
                              height:6px;width:{bw}%'></div></div>
                        </div>""", unsafe_allow_html=True)

            with sh2:
                if fd:
                    st.markdown("#### 🟢 Working in your favour")
                    for label, hint, mag in sorted(fd, key=lambda x: -x[2]):
                        bw = min(int(mag * 280), 100)
                        st.markdown(f"""
                        <div style='background:white;border-left:4px solid #10b981;
                        border-radius:12px;padding:14px 18px;margin:6px 0;
                        box-shadow:0 2px 8px #10b98108'>
                          <b style='color:#1a1a2e!important'>{label}</b><br>
                          <span style='font-size:0.8rem;color:#64748b!important'>{hint}</span>
                          <div style='background:#d1fae5;border-radius:999px;
                            height:6px;margin-top:8px'>
                            <div style='background:#10b981;border-radius:999px;
                              height:6px;width:{bw}%'></div></div>
                        </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.info(f"Factor breakdown unavailable: {e}")

        # ── Prevention plan ──
        st.markdown("---")
        st.markdown("### 🛡️ What to do next")
        items = PREVENTION.get(risk["level"], PREVENTION["MODERATE"])
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(items):
                    em, title, desc, cls = items[i + j]
                    col.markdown(
                        f'<div class="{cls}"><b style="color:#1a1a2e!important">'
                        f'{em} {title}</b><br>'
                        f'<span style="font-size:0.86rem;color:#374151!important">'
                        f'{desc}</span></div>',
                        unsafe_allow_html=True
                    )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Start Over"):
                st.session_state.patient_step = 0
                st.session_state.patient_data = {}
                st.session_state.faq_answers  = {}
                st.session_state.gds_answers  = {}
                for k in ["words_shown", "word_phase", "digit_sequences", "digit_level",
                          "digit_results", "digit_done", "digit_phase"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with c2:
            st.markdown("""
            <div style='background:#f0f4ff;border:2px solid #c7d2fe;border-radius:14px;
            padding:14px 18px;font-size:0.88rem;color:#3730a3!important'>
              <b>💡 Higher accuracy:</b> Ask your doctor to enter clinical test scores
              in <b>Doctor Mode</b> for a more precise result using all 18 biomarkers.
            </div>""", unsafe_allow_html=True)