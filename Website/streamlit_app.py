"""
streamlit_app.py — Memory Health Assessment Tool
Two modes: Doctor (score entry) | Patient (interactive tests)
Run: streamlit run streamlit_app.py

CHANGES v2:
-----------
CHANGE 1 — Comorbidity inputs added to Doctor Mode form
           (Diabetes, Hypertension, Smoking, BMI — §Medical History card)

CHANGE 2 — Three prediction threshold modes added to Doctor Mode
           Screening (t=0.15) | Balanced (t=0.35) | Confirmatory (t=0.50)

CHANGE 3 — Prediction now applies selected threshold — risk display
           reflects the chosen clinical use case.

CHANGE 4 — Landing page AUC updated to 0.805 [0.732–0.870] to match
           new scale_pos_weight model output.

CHANGE 5 — Comorbidity risk flags added to Doctor Mode results panel.
"""

import os, warnings, json, random
import numpy as np
import joblib, pickle
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_XGB = os.path.join(BASE_DIR, "models", "best_xgb.pkl")
MODEL_RF  = os.path.join(BASE_DIR, "models", "best_rf.pkl")
MODEL_LR  = os.path.join(BASE_DIR, "models", "best_lr.pkl")
MODEL_CNN = None   # CNN disabled for cloud deployment

st.set_page_config(page_title="Memory Assessment Tool",
                   page_icon="🧠", layout="wide",
                   initial_sidebar_state="collapsed")

# ════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, html, body { font-family: 'Sora', sans-serif !important; }
.stApp { background: #fafafa; }

.mode-doctor {
    background: linear-gradient(135deg, #0f172a 60%, #1e3a5f);
    border-radius: 24px; padding: 40px 36px; cursor: pointer;
    border: 3px solid transparent; transition: all .25s;
    color: white; text-align: center;
    box-shadow: 0 8px 40px #0f172a22;
}
.mode-patient {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
    border-radius: 24px; padding: 40px 36px; cursor: pointer;
    border: 3px solid transparent; transition: all .25s;
    color: white; text-align: center;
    box-shadow: 0 8px 40px #7c3aed22;
}
.mode-doctor:hover { transform:translateY(-6px); box-shadow:0 16px 50px #0f172a44; }
.mode-patient:hover { transform:translateY(-6px); box-shadow:0 16px 50px #7c3aed44; }

.doc-card {
    background: white; border-radius: 16px;
    border: 1px solid #e2e8f0;
    padding: 24px 28px; margin: 10px 0;
    box-shadow: 0 2px 16px #0001;
    border-left: 5px solid #1e3a5f;
}
/* CHANGE 1: Medical history card — orange left border */
.doc-card-med {
    background: #fff8f0; border-radius: 16px;
    border: 1px solid #fed7aa;
    padding: 24px 28px; margin: 10px 0;
    box-shadow: 0 2px 16px #0001;
    border-left: 5px solid #f97316;
}
/* CHANGE 2: Threshold mode selector card */
.threshold-card {
    background: #f0f9ff; border-radius: 16px;
    border: 1px solid #bae6fd;
    padding: 18px 24px; margin: 12px 0;
    border-left: 5px solid #0284c7;
}
/* CHANGE 5: Comorbidity risk flag */
.comorbidity-flag {
    background: #fff7ed; border-radius: 12px;
    border-left: 4px solid #f97316;
    padding: 12px 18px; margin: 6px 0;
    font-size: 0.87rem;
}

.pat-card {
    background: white; border-radius: 20px;
    border: 1px solid #ede9fe;
    padding: 28px 32px; margin: 12px 0;
    box-shadow: 0 4px 24px #7c3aed0f;
}
.pat-step-badge {
    display:inline-flex; align-items:center; justify-content:center;
    width:44px; height:44px; border-radius:50%;
    background: linear-gradient(135deg,#7c3aed,#db2777);
    color:white; font-weight:800; font-size:1.1rem;
    box-shadow: 0 4px 12px #7c3aed33; margin-right:12px;
}
.word-box {
    display:inline-block; background:#f5f3ff;
    border:2px solid #7c3aed; border-radius:12px;
    padding:10px 18px; margin:6px;
    font-size:1.15rem; font-weight:700; color:#4c1d95;
    font-family:'IBM Plex Mono', monospace;
}
.progress-track { background:#ede9fe; border-radius:999px; height:8px; margin:8px 0; }
.progress-fill {
    background:linear-gradient(90deg,#7c3aed,#db2777);
    border-radius:999px; height:8px; transition:width .4s ease;
}
.result-high   { background:linear-gradient(135deg,#fff1f2,#fef2f2); border:2px solid #fca5a5; border-radius:20px; padding:32px; text-align:center; }
.result-medium { background:linear-gradient(135deg,#fffbeb,#fefce8); border:2px solid #fcd34d; border-radius:20px; padding:32px; text-align:center; }
.result-low    { background:linear-gradient(135deg,#f0fdf4,#ecfdf5); border:2px solid #6ee7b7; border-radius:20px; padding:32px; text-align:center; }
.prev-urgent { background:white; border-left:5px solid #ef4444; border-radius:14px; padding:18px 22px; margin:8px 0; box-shadow:0 2px 12px #ef44440f; }
.prev-warn   { background:white; border-left:5px solid #f59e0b; border-radius:14px; padding:18px 22px; margin:8px 0; box-shadow:0 2px 12px #f59e0b0f; }
.prev-good   { background:white; border-left:5px solid #10b981; border-radius:14px; padding:18px 22px; margin:8px 0; box-shadow:0 2px 12px #10b9810f; }

.stButton>button {
    font-family:'Sora',sans-serif !important; font-weight:700;
    border-radius:14px; padding:12px 28px; width:100%;
    border:none; font-size:1rem; transition:all .2s;
}
.stButton>button:hover { transform:translateY(-2px); }
.digit-display {
    font-family:'IBM Plex Mono',monospace;
    font-size:3.5rem; font-weight:800; letter-spacing:16px;
    text-align:center; color:#1e293b; padding:24px;
    background:#f8fafc; border-radius:20px;
    border:3px solid #e2e8f0; margin:16px 0;
}
.nav-pill-active {
    background:linear-gradient(135deg,#7c3aed,#db2777);
    color:white; border-radius:999px; padding:6px 18px;
    font-weight:700; font-size:0.88rem; display:inline-block;
}
.nav-pill {
    background:#f1f5f9; color:#64748b;
    border-radius:999px; padding:6px 18px;
    font-weight:600; font-size:0.88rem; display:inline-block;
}
.header-bar {
    background: linear-gradient(90deg,#0f172a,#1e3a5f);
    border-radius: 20px; padding: 18px 28px;
    display:flex; align-items:center; gap:16px; margin-bottom:24px;
}
[data-testid="stNumberInput"] input {
    font-family:'IBM Plex Mono',monospace; font-size:1rem;
    background:#f8fafc; border:2px solid #e2e8f0; border-radius:10px;
}
[data-testid="stNumberInput"] input:focus { border-color:#7c3aed; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    m = {}
    for name, path in [("XGBoost", MODEL_XGB),
                        ("Random Forest", MODEL_RF),
                        ("Logistic Regression", MODEL_LR)]:
        if not os.path.exists(path):
            continue
        try:
            m[name] = joblib.load(path)
        except:
            try:
                with open(path, "rb") as f:
                    m[name] = pickle.load(f)
            except:
                pass
    return m

@st.cache_resource(show_spinner=False)
def load_cnn():
    try:
        import torch, torchvision.models as tvm, torch.nn as nn
        mdl = tvm.resnet18(weights=None)
        mdl.fc = nn.Linear(512, 2)
        ckpt = torch.load(MODEL_CNN, map_location="cpu")
        mdl.load_state_dict(ckpt["model_state_dict"])
        mdl.eval()
        return mdl
    except:
        return None

models     = load_models()
cnn_model  = load_cnn()
primary    = models.get("XGBoost") or (list(models.values())[0] if models else None)

ALL_FEATURES = [
    "PTGENDER", "PTEDUCAT", "MMSE_BL", "MOCA_BL", "ADAS11_BL",
    "ADAS13_BL", "FAQ_BL", "GDS_BL", "CDR_GLOBAL_BL", "CDRSB_BL",
    "RAVLT_forgetting", "RAVLT_immediate", "RAVLT_delayed", "DigitSpan",
    "TrailsB", "MMSE_FAQ_composite", "ADAS_MMSE_gap", "RAVLT_forget_rate",
]

WORD_LIST = [
    "drum", "curtain", "bell", "coffee", "school", "parent",
    "moon", "garden", "hat", "farmer", "nose", "turkey", "river", "house", "road",
]

DIGIT_SEQUENCES = [
    [5,8,2], [6,9,4,1], [7,4,2,8,5], [3,9,1,7,4,2], [8,1,5,9,3,6,4],
    [5,2,8,1,4,7,9,3], [9,3,7,1,5,4,8,2,6],
]

FAQ_QUESTIONS = [
    "Writing cheques or paying bills",
    "Shopping alone",
    "Heating water, making coffee or turning off appliances",
    "Preparing a balanced meal",
    "Keeping track of current events",
    "Watching a TV series and following the plot",
    "Remembering appointments or family occasions",
    "Driving or using public transport",
    "Filling in forms or doing paperwork",
    "Managing medicines (correct dose at correct time)",
]
GDS_QUESTIONS = [
    ("Are you basically satisfied with your life?",                "no"),
    ("Have you dropped many of your activities?",                  "yes"),
    ("Do you feel your life is empty?",                            "yes"),
    ("Do you often get bored?",                                    "yes"),
    ("Are you in good spirits most of the time?",                  "no"),
    ("Are you afraid that something bad is going to happen?",      "yes"),
    ("Do you feel happy most of the time?",                        "no"),
    ("Do you often feel helpless?",                                "yes"),
    ("Do you prefer to stay at home rather than go out?",          "yes"),
    ("Do you feel you have more problems with memory than most?",  "yes"),
    ("Do you think it is wonderful to be alive now?",              "no"),
    ("Do you feel worthless the way you are now?",                 "yes"),
    ("Do you feel full of energy?",                                "no"),
    ("Do you feel your situation is hopeless?",                    "yes"),
    ("Do you think that most people are better off than you?",     "yes"),
]

# ── CHANGE 2: Threshold map ───────────────────────────────────────────────────
THRESHOLD_OPTIONS = {
    "🔍 Screening  (t=0.15 — high sensitivity, catches most cases)":  0.15,
    "⚖️ Balanced   (t=0.35 — recommended for general clinical use)":   0.35,
    "✅ Confirmatory (t=0.50 — high specificity, fewer false alarms)": 0.50,
}

def compute_composites(v):
    v["MMSE_FAQ_composite"] = float(v["MMSE_BL"]) - float(v["FAQ_BL"])
    v["ADAS_MMSE_gap"]      = float(v["ADAS13_BL"]) + (30 - float(v["MMSE_BL"]))
    avg = max(float(v["RAVLT_immediate"]) / 5, 0.1)
    v["RAVLT_forget_rate"]  = float(v["RAVLT_forgetting"]) / avg
    return v

def risk_info(prob, threshold=0.50):
    # CHANGE 3: threshold-aware risk tier boundaries
    # Screening (0.15): boundaries 0.40 / 0.65
    # Balanced  (0.35): boundaries 0.50 / 0.70
    # Confirmatory (0.50): boundaries 0.55 / 0.75
    if threshold <= 0.15:
        lo, hi = 0.40, 0.65
    elif threshold <= 0.35:
        lo, hi = 0.50, 0.70
    else:
        lo, hi = 0.55, 0.75

    if prob >= hi:
        return {"level": "HIGH", "color": "#ef4444", "cls": "result-high",
                "emoji": "⚠️", "pct": int(prob * 100),
                "headline": "Higher risk of memory decline",
                "plain": "The scores suggest a higher-than-average chance of "
                         "memory getting worse in the next 3 years. "
                         "Please consult a neurologist soon."}
    if prob >= lo:
        return {"level": "MODERATE", "color": "#f59e0b", "cls": "result-medium",
                "emoji": "🔶", "pct": int(prob * 100),
                "headline": "Moderate — keep a close watch",
                "plain": "There are some warning signs. Not immediately alarming, "
                         "but worth monitoring carefully with regular doctor visits."}
    return {"level": "LOWER", "color": "#10b981", "cls": "result-low",
            "emoji": "✅", "pct": int(prob * 100),
            "headline": "Lower risk at this time",
            "plain": "Fewer signs of progression right now. Continue healthy "
                     "habits and annual check-ups."}

PREVENTION = {
    "HIGH": [
        ("🏥 See a neurologist soon",
         "Book an appointment this week. Early treatment makes the biggest difference.",
         "prev-urgent"),
        ("💊 Medication review",
         "Some medicines affect memory. Ask your doctor to review all current medications.",
         "prev-urgent"),
        ("🥗 Mediterranean diet",
         "Fish, olive oil, vegetables, nuts, whole grains. Reduce sugar and processed food.",
         "prev-warn"),
        ("🏃 Exercise 30 min daily",
         "Walking, swimming or cycling improves blood flow to the brain.",
         "prev-warn"),
        ("😴 Fix sleep quality",
         "7–8 hours every night. Treat snoring or insomnia with medical help.",
         "prev-warn"),
        ("🧩 Keep the mind active",
         "Reading, puzzles, learning something new every day builds cognitive reserve.",
         "prev-warn"),
        ("❤️ Control BP, diabetes & cholesterol",
         "These directly increase dementia risk — get them checked.",
         "prev-warn"),
        ("🚭 Reduce alcohol & quit smoking",
         "Both damage brain cells significantly.",
         "prev-good"),
        ("👥 Stay socially connected",
         "Regular contact with family and friends reduces isolation-related decline.",
         "prev-good"),
    ],
    "MODERATE": [
        ("📅 Book a memory check-up",
         "Annual cognitive screening with a doctor is recommended.",
         "prev-warn"),
        ("🏃 Start exercising regularly",
         "Even a 20-minute daily walk makes a measurable difference.",
         "prev-warn"),
        ("🥗 Improve diet quality",
         "More fish, vegetables, and nuts. Less fried food and sugary drinks.",
         "prev-good"),
        ("🧩 Daily mental stimulation",
         "Crosswords, reading, or learning a new hobby.",
         "prev-good"),
        ("😴 Better sleep habits",
         "Consistent sleep and wake times. No screens 1 hour before bed.",
         "prev-good"),
        ("👥 Stay socially active",
         "Loneliness is a significant risk factor for cognitive decline.",
         "prev-good"),
    ],
    "LOWER": [
        ("✅ Maintain healthy habits",
         "Current scores are reassuring — keep up what you're doing.",
         "prev-good"),
        ("🏃 Stay physically active",
         "Exercise remains the most powerful protection against cognitive decline.",
         "prev-good"),
        ("📅 Annual health check-ups",
         "Blood pressure, blood sugar, and cholesterol check every year.",
         "prev-good"),
        ("🧩 Keep challenging your brain",
         "New experiences and learning build long-term cognitive reserve.",
         "prev-good"),
    ],
}

# ── CHANGE 5: Comorbidity risk messages ──────────────────────────────────────
def comorbidity_flags(diabetes, hypertension, smoking, bmi):
    flags = []
    if diabetes == "Yes":
        flags.append("🩸 <b>Type 2 Diabetes</b> — insulin resistance is directly linked "
                     "to amyloid accumulation. Tight glycaemic control is important.")
    if hypertension == "Yes":
        flags.append("💉 <b>Hypertension</b> — vascular damage accelerates cognitive "
                     "decline. Ensure BP is regularly monitored and treated.")
    if smoking == "Current":
        flags.append("🚬 <b>Current smoker</b> — smoking is a significant cerebrovascular "
                     "risk factor. Cessation support is strongly recommended.")
    elif smoking == "Former":
        flags.append("🚬 <b>Former smoker</b> — residual cerebrovascular risk present. "
                     "Continue smoke-free lifestyle.")
    if bmi >= 30:
        flags.append(f"⚖️ <b>BMI {bmi:.1f} (Obese)</b> — midlife obesity significantly "
                     "increases dementia risk. Weight management advised.")
    elif bmi >= 25:
        flags.append(f"⚖️ <b>BMI {bmi:.1f} (Overweight)</b> — some elevated risk. "
                     "Healthy diet and exercise are recommended.")
    return flags

# ════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ════════════════════════════════════════════════════════
if "mode" not in st.session_state:
    st.session_state.mode = None
if "patient_step" not in st.session_state:
    st.session_state.patient_step = 0
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

# ════════════════════════════════════════════════════════
# LANDING PAGE — MODE SELECTOR
# ════════════════════════════════════════════════════════
if st.session_state.mode is None:
    st.markdown("""
    <div style='text-align:center; padding:40px 0 20px'>
        <div style='font-size:3.5rem'>🧠</div>
        <h1 style='font-size:2.8rem; font-weight:800; color:#0f172a; margin:8px 0'>
            Memory Health Assessment</h1>
        <p style='color:#64748b; font-size:1.1rem; max-width:560px; margin:0 auto 8px'>
            AI-powered tool to assess the risk of MCI progressing to dementia.</p>
        <div style='background:#fffbeb;border:1px solid #fcd34d;border-left:4px solid #f59e0b;
            border-radius:12px;padding:10px 20px;display:inline-block;
            font-size:0.85rem;color:#92400e;margin-top:12px'>
        ⚠️ Research prototype. Not for clinical diagnosis. Always consult a qualified doctor.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Choose your mode")
    st.markdown("")

    c1, gap, c2 = st.columns([1, 0.08, 1])
    with c1:
        st.markdown("""
        <div class="mode-doctor">
            <div style='font-size:3.5rem; margin-bottom:12px'>👨‍⚕️</div>
            <h2 style='margin:0 0 8px; font-size:1.8rem'>Doctor / Clinician</h2>
            <p style='opacity:0.8; margin:0; font-size:0.95rem; line-height:1.6'>
            Enter clinical test scores directly.<br>
            Immediate prediction with SHAP analysis<br>
            and ensemble model comparison.
            </p>
            <div style='margin-top:20px; background:rgba(255,255,255,0.15);
                border-radius:10px; padding:8px 16px; font-size:0.83rem'>
            ⚡ Fast • Precise • All 18 clinical features • 3 threshold modes
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Enter Doctor Mode →", key="btn_doc"):
            st.session_state.mode = "doctor"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="mode-patient">
            <div style='font-size:3.5rem; margin-bottom:12px'>🧑‍🤝‍🧑</div>
            <h2 style='margin:0 0 8px; font-size:1.8rem'>Patient / Family</h2>
            <p style='opacity:0.9; margin:0; font-size:0.95rem; line-height:1.6'>
            Answer simple questions and do<br>
            interactive memory tests — the app<br>
            calculates scores automatically.
            </p>
            <div style='margin-top:20px; background:rgba(255,255,255,0.2);
                border-radius:10px; padding:8px 16px; font-size:0.83rem'>
            🎯 Interactive • Plain language • Voice memory test
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Enter Patient Mode →", key="btn_pat"):
            st.session_state.mode = "patient"
            st.session_state.patient_step = 0
            st.rerun()

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        # CHANGE 4: updated AUC to 0.805 [0.732–0.870]
        st.markdown("""<div style='text-align:center;padding:16px'>
        <div style='font-size:2rem'>🤖</div>
        <b>AI-powered</b><br>
        <span style='font-size:0.85rem;color:#64748b'>
        Trained on 767 ADNI patients<br>AUC-ROC: 0.805 [0.732–0.870]</span>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div style='text-align:center;padding:16px'>
        <div style='font-size:2rem'>🔒</div>
        <b>Private & local</b><br>
        <span style='font-size:0.85rem;color:#64748b'>
        All data stays on your computer.<br>Nothing is uploaded.</span>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""<div style='text-align:center;padding:16px'>
        <div style='font-size:2rem'>🧩</div>
        <b>Interactive tests</b><br>
        <span style='font-size:0.85rem;color:#64748b'>
        Memory, digit span, and<br>daily function assessments.</span>
        </div>""", unsafe_allow_html=True)

    st.stop()

# ════════════════════════════════════════════════════════
# HEADER (both modes)
# ════════════════════════════════════════════════════════
mode_label = "👨‍⚕️ Doctor Mode" if st.session_state.mode == "doctor" else "🧑‍🤝‍🧑 Patient Mode"
mode_color = "#1e3a5f"  if st.session_state.mode == "doctor" else "#7c3aed"
h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(f"""<div style='background:linear-gradient(90deg,{mode_color},
        {"#2563eb" if st.session_state.mode=="doctor" else "#db2777"});
        border-radius:16px;padding:14px 24px;color:white;margin-bottom:20px'>
        <span style='font-size:1.1rem;font-weight:700'>🧠 Memory Assessment Tool</span>
        <span style='margin-left:20px;opacity:0.8'>|  {mode_label}</span>
    </div>""", unsafe_allow_html=True)
with h2:
    if st.button("← Switch Mode"):
        st.session_state.mode = None
        st.session_state.patient_step = 0
        st.session_state.patient_data = {}
        for k in ["clin_prob", "mri_prob", "risk", "vals", "words_shown",
                  "immediate_recall", "delayed_recall", "digit_score",
                  "selected_threshold"]:   # CHANGE 2: clear threshold too
            st.session_state.pop(k, None)
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# DOCTOR MODE
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "doctor":
    tab1, tab2, tab3 = st.tabs(["📋 Assessment Form", "📊 Results & SHAP", "🧲 MRI Upload"])

    with tab1:
        st.markdown("#### Enter clinical test scores")
        st.markdown('<div style="font-size:0.88rem;color:#64748b;margin-bottom:16px">'
                    "Fill in scores from the patient's cognitive assessment report.</div>",
                    unsafe_allow_html=True)
        if not models:
            st.error(f"No models found at `{DATA_ROOT}`")
            st.stop()

        with st.form("doctor_form"):
            c1, c2, c3 = st.columns(3)

            # ── Column 1: Demographics + Medical History ──────────────────────
            with c1:
                st.markdown('<div class="doc-card"><b>👤 Demographics</b></div>',
                            unsafe_allow_html=True)
                sex   = st.selectbox("Sex", ["Male", "Female"])
                edu   = st.number_input("Education (years)", 0, 25, 14, 1,
                                         help="0 = no formal education")
                cdr_g = st.selectbox("CDR Global", [0.0, 0.5, 1.0, 2.0, 3.0], index=1)
                cdrsb = st.number_input("CDR Sum of Boxes", 0.0, 18.0, 1.5, 0.5)

                # ── CHANGE 1: Medical History section ────────────────────────
                st.markdown('<div class="doc-card-med"><b>🏥 Medical History</b></div>',
                            unsafe_allow_html=True)
                diabetes     = st.selectbox("Type 2 Diabetes",
                                            ["No", "Yes"], index=0,
                                            help="Insulin resistance linked to amyloid accumulation")
                hypertension = st.selectbox("Hypertension",
                                            ["No", "Yes"], index=0,
                                            help="Vascular damage accelerates cognitive decline")
                smoking      = st.selectbox("Smoking History",
                                            ["Never", "Former", "Current"], index=0)
                bmi          = st.number_input("BMI",
                                               min_value=15.0, max_value=50.0,
                                               value=25.0, step=0.1,
                                               help="BMI ≥ 30 (obese) increases dementia risk")

            # ── Column 2: Memory + Function ───────────────────────────────────
            with c2:
                st.markdown('<div class="doc-card"><b>🧠 Memory (RAVLT)</b></div>',
                            unsafe_allow_html=True)
                rv_imm = st.number_input("RAVLT Immediate (0–75)", 0.0, 75.0, 35.0, 0.5)
                rv_del = st.number_input("RAVLT Delayed (0–15)",   0.0, 15.0,  7.0, 0.5)
                rv_fo  = st.number_input("RAVLT Forgetting (0–15)",0.0, 15.0,  4.0, 0.5)
                ds     = st.number_input("Digit Span (0–28)",       0.0, 28.0, 14.0, 0.5)
                st.markdown('<div class="doc-card"><b>🏠 Function</b></div>',
                            unsafe_allow_html=True)
                faq    = st.number_input("FAQ (0–30)", 0.0, 30.0,  5.0, 0.5)
                gds    = st.number_input("GDS (0–15)", 0.0, 15.0,  2.0, 0.5)

            # ── Column 3: Cognition ───────────────────────────────────────────
            with c3:
                st.markdown('<div class="doc-card"><b>💭 Cognition</b></div>',
                            unsafe_allow_html=True)
                mmse   = st.number_input("MMSE (0–30)",          0.0, 30.0, 26.0, 0.5)
                moca   = st.number_input("MoCA (0–30)",          0.0, 30.0, 23.0, 0.5)
                adas13 = st.number_input("ADAS-Cog 13",          0.0, 85.0, 18.0, 0.5)
                adas11 = st.number_input("ADAS-Cog 11",          0.0, 70.0, 13.0, 0.5)
                trails = st.number_input("Trails B (seconds)",  10.0,300.0,120.0, 5.0)

            # ── CHANGE 2: Threshold mode selector ────────────────────────────
            st.markdown('<div class="threshold-card">', unsafe_allow_html=True)
            st.markdown("**🎯 Prediction Mode** — choose clinical use case:")
            threshold_mode = st.radio(
                "Threshold",
                list(THRESHOLD_OPTIONS.keys()),
                index=1,                       # default: Balanced (t=0.35)
                label_visibility="collapsed",
                help="Screening catches more at-risk patients but flags more "
                     "false positives. Confirmatory is stricter.",
            )
            st.markdown(
                "<div style='font-size:0.82rem;color:#0369a1;margin-top:4px'>"
                f"Selected threshold: <b>t = {THRESHOLD_OPTIONS[threshold_mode]:.2f}</b>  "
                "— from xgb_threshold_tuning.csv"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            mri_w     = st.slider("Fusion weight (if MRI available)", 0.0, 0.5, 0.0, 0.05,
                                   help="0 = clinical only. Set > 0 after uploading MRI in Tab 3.")
            submitted = st.form_submit_button("🔍  Predict Now", use_container_width=True)

        if submitted:
            # Build feature vector
            vals = {
                "PTGENDER":    0.0 if sex == "Male" else 1.0,
                "PTEDUCAT":    float(edu),
                "MMSE_BL":     mmse,
                "MOCA_BL":     moca,
                "ADAS11_BL":   adas11,
                "ADAS13_BL":   adas13,
                "FAQ_BL":      faq,
                "GDS_BL":      gds,
                "CDR_GLOBAL_BL": cdr_g,
                "CDRSB_BL":    cdrsb,
                "RAVLT_forgetting": rv_fo,
                "RAVLT_immediate":  rv_imm,
                "RAVLT_delayed":    rv_del,
                "DigitSpan":   ds,
                "TrailsB":     trails,
            }
            vals = compute_composites(vals)
            st.session_state.vals = vals

            X          = np.array([[vals[f] for f in ALL_FEATURES]])
            clin_prob  = float(primary.predict_proba(X)[0, 1])
            mri_prob   = st.session_state.get("mri_prob", None)

            if mri_prob and mri_w > 0:
                final_prob = (1 - mri_w) * clin_prob + mri_w * mri_prob
            else:
                final_prob = clin_prob

            # CHANGE 2 & 3: store threshold and compute risk with it
            threshold = THRESHOLD_OPTIONS[threshold_mode]
            st.session_state.clin_prob          = clin_prob
            st.session_state.selected_threshold = threshold
            st.session_state.risk               = risk_info(final_prob, threshold)

            # CHANGE 1: store comorbidity inputs for Results tab
            st.session_state.comorbidities = {
                "diabetes":     diabetes,
                "hypertension": hypertension,
                "smoking":      smoking,
                "bmi":          bmi,
            }

            st.success("✅ Prediction complete — go to **📊 Results & SHAP** tab")

    # ── Tab 2: Results & SHAP ─────────────────────────────────────────────────
    with tab2:
        risk       = st.session_state.get("risk")
        vals       = st.session_state.get("vals")
        clin_prob  = st.session_state.get("clin_prob")
        threshold  = st.session_state.get("selected_threshold", 0.35)  # CHANGE 2
        comorbids  = st.session_state.get("comorbidities", {})          # CHANGE 1

        if risk is None:
            st.info("Complete the assessment form first.")
        else:
            r1, r2 = st.columns([1, 1])
            with r1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk["pct"],
                    title={"text": "Conversion Risk (%)", "font": {"size": 14, "color": "#374151"}},
                    number={"suffix": "%", "font": {"size": 44, "color": risk["color"]}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": risk["color"], "thickness": 0.28},
                        "bgcolor": "#f8fafc",
                        "steps": [
                            {"range": [0,  35], "color": "#f0fdf4"},
                            {"range": [35, 65], "color": "#fffbeb"},
                            {"range": [65,100], "color": "#fef2f2"},
                        ],
                        "threshold": {"line": {"color": "#94a3b8", "width": 2}, "value": 50},
                    },
                ))
                fig.update_layout(height=260, paper_bgcolor="white",
                                  margin=dict(t=50, b=10, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

                # CHANGE 2: show which threshold was used
                st.markdown(
                    f"<div style='text-align:center;font-size:0.82rem;color:#0369a1'>"
                    f"Threshold used: <b>t = {threshold:.2f}</b></div>",
                    unsafe_allow_html=True,
                )

            with r2:
                st.markdown(f"""<div class="{risk['cls']}">
                <div style='font-size:3rem;margin-bottom:8px'>{risk['emoji']}</div>
                <div style='font-size:1.4rem;font-weight:800;color:{risk["color"]}'>{risk['headline']}</div>
                <div style='color:#374151;font-size:0.92rem;margin-top:10px;line-height:1.7'>{risk['plain']}</div>
                </div>""", unsafe_allow_html=True)

                if len(models) > 1:
                    st.markdown("**Ensemble comparison:**")
                    for nm, mdl in models.items():
                        X2 = np.array([[vals[f] for f in ALL_FEATURES]])
                        mp = float(mdl.predict_proba(X2)[0, 1])
                        st.metric(nm, f"{mp:.3f}",
                                  delta="▲ pMCI" if mp >= threshold else "▼ sMCI")

            # CHANGE 5: Comorbidity risk flags
            if comorbids:
                flags = comorbidity_flags(
                    comorbids.get("diabetes", "No"),
                    comorbids.get("hypertension", "No"),
                    comorbids.get("smoking", "Never"),
                    comorbids.get("bmi", 25.0),
                )
                if flags:
                    st.markdown("#### 🏥 Comorbidity Risk Factors")
                    st.markdown(
                        "<div style='font-size:0.84rem;color:#64748b;margin-bottom:8px'>"
                        "These factors are independent risk amplifiers not captured "
                        "by the cognitive model alone.</div>",
                        unsafe_allow_html=True,
                    )
                    for flag in flags:
                        st.markdown(
                            f'<div class="comorbidity-flag">{flag}</div>',
                            unsafe_allow_html=True,
                        )

            # SHAP waterfall
            if vals and primary:
                st.markdown("#### SHAP Feature Importance")
                try:
                    X = np.array([[vals[f] for f in ALL_FEATURES]])
                    if hasattr(primary, "steps"):
                        from sklearn.pipeline import Pipeline as SKPipeline
                        xgb_model2 = primary.steps[-1][1]
                        if len(primary.steps) > 1:
                            pre2 = SKPipeline(primary.steps[:-1])
                            try:    X_s2 = pre2.transform(X)
                            except: X_s2 = X
                        else:
                            X_s2 = X
                    else:
                        xgb_model2 = primary
                        X_s2       = X

                    exp = shap.TreeExplainer(xgb_model2)
                    sv  = exp.shap_values(X_s2)[0]
                    idx = np.argsort(np.abs(sv))[-12:][::-1]
                    nm_ = [f.replace("_BL", "").replace("_", " ") for f in ALL_FEATURES]
                    top_nm = [nm_[i] for i in idx]
                    top_sv = sv[idx]

                    fig2, ax = plt.subplots(figsize=(7, 4))
                    fig2.patch.set_facecolor("white")
                    ax.set_facecolor("#f8fafc")
                    ax.barh(top_nm[::-1], top_sv[::-1],
                            color=["#ef4444" if v > 0 else "#3b82f6" for v in top_sv[::-1]],
                            alpha=0.85, height=0.65)
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

    # ── Tab 3: MRI Upload ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Upload Brain MRI (NIfTI)")
        cnn_status = "✅ CNN model loaded" if cnn_model else "⚠️ CNN model not found"
        st.caption(cnn_status)
        uploaded = st.file_uploader("Upload .nii or .nii.gz", type=["nii", "gz"])
        if uploaded:
            with st.spinner("Processing MRI…"):
                try:
                    import nibabel as nib, torch
                    from PIL import Image
                    import torchvision.transforms as T

                    tmp = os.path.join(os.environ.get("TEMP", "C:\\Temp"), uploaded.name)
                    with open(tmp, "wb") as f:
                        f.write(uploaded.read())
                    d = np.array(nib.load(tmp).get_fdata(), dtype=np.float32)
                    if d.ndim == 4:
                        d = d[..., 0]
                    nz = d.shape[2]; mid = int(nz * .5)
                    sls, prev = [], None
                    for off in [-1, 0, 1]:
                        sl = d[:, :, np.clip(mid + off, 0, nz - 1)]
                        lo, hi = np.percentile(sl, [1, 99])
                        sl = np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)
                        u8 = (sl * 255).astype(np.uint8)
                        if off == 0: prev = u8
                        sls.append(u8)
                    if cnn_model:
                        from PIL import Image
                        rgb = np.stack(sls, 2); pil = Image.fromarray(rgb, "RGB")
                        t = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                       T.Normalize([.485,.456,.406],[.229,.224,.225])])
                        with torch.no_grad():
                            mp = float(torch.softmax(cnn_model(t(pil).unsqueeze(0)), 1)[0, 1])
                        st.session_state.mri_prob = mp
                        os.remove(tmp)
                        c1, c2 = st.columns(2)
                        with c1:
                            fig3, ax = plt.subplots(figsize=(4, 4))
                            ax.imshow(prev, cmap="gray"); ax.axis("off")
                            st.pyplot(fig3, use_container_width=True); plt.close(fig3)
                        with c2:
                            ri = risk_info(mp)
                            st.metric("CNN MRI Risk", f"{int(mp * 100)}%")
                            st.markdown(f"**{ri['emoji']} {ri['headline']}**")
                        st.success("MRI probability saved. Set fusion weight > 0 in the form.")
                    else:
                        st.warning("CNN model not loaded.")
                except Exception as e:
                    st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════════
# PATIENT MODE — Step-by-step wizard
# ════════════════════════════════════════════════════════════════════════════════
else:
    step  = st.session_state.patient_step
    pd_   = st.session_state.patient_data

    STEPS = ["Personal Info", "Word Memory Test", "Daily Life Questions",
             "Mood Questions", "Digit Span Test", "Word Recall", "Results"]
    total = len(STEPS)
    pct   = int((step / max(total - 1, 1)) * 100)

    pills = "".join([
        f'<span class="{"nav-pill-active" if i == step else "nav-pill"}">{s}</span> '
        for i, s in enumerate(STEPS)
    ])
    st.markdown(f"""
    <div style='margin-bottom:8px'>{pills}</div>
    <div class='progress-track'><div class='progress-fill' style='width:{pct}%'></div></div>
    <div style='font-size:0.78rem;color:#64748b;text-align:right;margin-bottom:16px'>
    Step {step + 1} of {total}</div>""", unsafe_allow_html=True)

    # ── STEP 0: Personal Info ─────────────────────────────────────────────────
    if step == 0:
        st.markdown("## 👤 Tell us about the person")
        with st.form("p0"):
            st.markdown('<div class="pat-card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("First name (optional)", placeholder="e.g. Ravi")
                sex  = st.selectbox("Sex at birth",
                                    ["Male", "Female", "Other / Prefer not to say"])
            with c2:
                edu = st.slider("Years of school/college", 0, 25, 10,
                                 help="0 is perfectly fine — the AI accounts for this")
                if edu == 0:
                    st.caption("✅ No formal education — accounted for in the model")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("""<div style='background:#eff6ff;border:1px solid #bfdbfe;
            border-radius:10px;padding:10px 16px;font-size:0.84rem;color:#1e40af;margin-top:8px'>
            ℹ️ <b>CDR and other clinical scores will be estimated automatically</b>
            from the tests you complete in the next steps.
            </div>""", unsafe_allow_html=True)
            if st.form_submit_button("Continue →", use_container_width=True):
                pd_["name"] = name; pd_["sex"] = sex; pd_["edu"] = edu
                st.session_state.patient_step = 1
                st.session_state.patient_data = pd_
                for k in ["word_phase", "digit_phase", "current_digit_seq",
                          "digit_level", "digit_correct", "digit_done"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # ── STEP 1: Word Memory Test ──────────────────────────────────────────────
    elif step == 1:
        name     = pd_.get("name", "")
        greeting = f"Hello {name}! " if name else ""

        if "word_phase" not in st.session_state:
            st.session_state.word_phase  = "show"
            st.session_state.words_shown = WORD_LIST[:]

        words    = st.session_state.words_shown
        words_js = json.dumps(words)

        if st.session_state.word_phase == "show":
            st.markdown("## 🧠 Word Memory Test — Read & Listen")
            st.markdown(f"*{greeting}We will show you **{len(words)} words**. "
                        "Read them carefully and use the button to hear them. "
                        "**Once you click 'Ready to Recall', the words will disappear** "
                        "and you must type what you remember.*")

            word_html = "".join([
                f'<span style="display:inline-block;background:#f5f3ff;border:2px solid #7c3aed;'
                f'border-radius:12px;padding:10px 18px;margin:6px;font-size:1.15rem;'
                f'font-weight:700;color:#4c1d95;font-family:monospace">{w.upper()}</span>'
                for w in words])
            st.markdown(f"""<div style='background:white;border-radius:20px;
            border:1px solid #ede9fe;padding:28px 32px;margin:12px 0;
            box-shadow:0 4px 24px #7c3aed0f'>
            <h3 style='margin:0 0 16px'>👀 Study these words carefully:</h3>
            <div style='margin:0 0 8px'>{word_html}</div></div>""",
            unsafe_allow_html=True)

            components.html(f"""
            <div id="controls" style="font-family:sans-serif;margin:8px 0;
                display:flex;align-items:center;gap:12px;flex-wrap:wrap">
              <button id="speakBtn" onclick="startSpeech()"
                style="background:linear-gradient(135deg,#7c3aed,#db2777);color:white;
                       border:none;border-radius:12px;padding:12px 22px;font-size:0.95rem;
                       font-weight:700;cursor:pointer">
                🔊 Listen to All Words</button>
              <button id="stopBtn" onclick="stopSpeech()" disabled
                style="background:#ef4444;color:white;border:none;border-radius:12px;
                       padding:12px 18px;font-size:0.95rem;font-weight:700;
                       cursor:pointer;opacity:0.4">⏹ Stop</button>
              <span id="status" style="color:#64748b;font-size:0.88rem"></span>
            </div>
            <div id="countdown-bar" style="display:none;margin-top:10px">
              <div style="font-size:0.85rem;color:#7c3aed;margin-bottom:4px">
                ⏱ Study time: <b id="timer">60</b>s remaining</div>
              <div style="background:#ede9fe;border-radius:999px;height:6px">
                <div id="bar" style="background:linear-gradient(90deg,#7c3aed,#db2777);
                     border-radius:999px;height:6px;width:100%;
                     transition:width 1s linear"></div>
              </div>
            </div>
            <script>
            const words = {words_js};
            let speaking = false; let cdTimer = null; let timeLeft = 60;
            function startSpeech() {{
              window.speechSynthesis.cancel(); speaking = true;
              document.getElementById('speakBtn').disabled = true;
              document.getElementById('stopBtn').disabled = false;
              document.getElementById('stopBtn').style.opacity = '1';
              document.getElementById('status').innerText = 'Playing...';
              document.getElementById('countdown-bar').style.display = 'block';
              startCountdown();
              let idx = 0;
              function next() {{
                if(!speaking || idx >= words.length) {{
                  document.getElementById('status').innerText = speaking ? '✅ Done!' : '⏹ Stopped.';
                  document.getElementById('speakBtn').disabled = false;
                  document.getElementById('speakBtn').innerText = '🔊 Listen Again';
                  document.getElementById('stopBtn').disabled = true;
                  document.getElementById('stopBtn').style.opacity = '0.4'; return;
                }}
                document.getElementById('status').innerText = 'Word ' + (idx+1) + '/' + words.length + ':  ' + words[idx].toUpperCase();
                const u = new SpeechSynthesisUtterance(words[idx]);
                u.rate = 0.8; u.onend = () => {{ idx++; setTimeout(next, 500); }};
                window.speechSynthesis.speak(u);
              }}
              next();
            }}
            function stopSpeech() {{
              speaking = false; window.speechSynthesis.cancel();
              document.getElementById('speakBtn').disabled = false;
              document.getElementById('speakBtn').innerText = '🔊 Listen Again';
              document.getElementById('stopBtn').disabled = true;
              document.getElementById('stopBtn').style.opacity = '0.4';
              document.getElementById('status').innerText = '⏹ Stopped.';
            }}
            function startCountdown() {{
              if(cdTimer) clearInterval(cdTimer); timeLeft = 60;
              cdTimer = setInterval(() => {{
                timeLeft--;
                document.getElementById('timer').innerText = timeLeft;
                document.getElementById('bar').style.width = (timeLeft/60*100) + '%';
                if(timeLeft <= 0) {{ clearInterval(cdTimer); document.getElementById('status').innerText = '⏰ Time up!'; }}
              }}, 1000);
            }}
            </script>""", height=120)

            st.warning("📌 **Study the words above carefully. Once you click below, "
                       "the words will be hidden.**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back"):
                    st.session_state.patient_step = 0; st.rerun()
            with col2:
                if st.button("✅  I'm Ready — Hide Words & Start Recall",
                             use_container_width=True):
                    st.session_state.word_phase = "recall"; st.rerun()

        else:
            st.markdown("## ✍️ Now Recall the Words")
            st.markdown("*The words are now hidden. Type every word you can remember.*")
            st.markdown('<div class="pat-card">', unsafe_allow_html=True)
            st.markdown("**Type the words you remember — separated by spaces or commas:**")
            immediate_input = st.text_area("Your answer:", height=100,
                                            key="imm_recall", placeholder="")
            st.markdown('</div>', unsafe_allow_html=True)
            st.info("💡 Don't worry if you can't remember all of them — just write what you can.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Go back and study again"):
                    st.session_state.word_phase = "show"; st.rerun()
            with col2:
                if st.button("Submit & Continue →", use_container_width=True):
                    recalled    = [w.strip().lower()
                                   for w in immediate_input.replace(",", " ").split()
                                   if w.strip()]
                    words_set   = set(w.lower() for w in words)
                    correct_imm = [w for w in recalled if w in words_set]
                    pd_["immediate_recall"] = correct_imm
                    pd_["rv_imm"]           = min(len(correct_imm) * 3.5, 75)
                    st.session_state.patient_step = 2
                    st.session_state.patient_data = pd_; st.rerun()

    # ── STEP 2: Daily Life Questions (FAQ) ────────────────────────────────────
    elif step == 2:
        st.markdown("## 🏠 Daily Life Questions")
        st.markdown("*For each activity, how much help does the person need?*")
        faq_score = 0
        options   = ["No help needed", "Sometimes needs help",
                     "Often needs help", "Cannot do it at all"]
        with st.form("faq_form"):
            for q in FAQ_QUESTIONS:
                resp = st.selectbox(f"**{q}**", options, key=f"faq_{q}")
                faq_score += options.index(resp)
            c1, c2 = st.columns(2)
            with c1:
                if st.form_submit_button("← Back"):
                    st.session_state.patient_step = 1; st.rerun()
            with c2:
                if st.form_submit_button("Continue →", use_container_width=True):
                    if faq_score <= 2:    cdr_g, cdrsb = 0.0,  0.0
                    elif faq_score <= 6:  cdr_g, cdrsb = 0.5,  2.0
                    elif faq_score <= 14: cdr_g, cdrsb = 1.0,  4.5
                    elif faq_score <= 22: cdr_g, cdrsb = 2.0,  9.0
                    else:                 cdr_g, cdrsb = 3.0, 14.0
                    pd_["faq"]   = faq_score
                    pd_["cdr_g"] = cdr_g
                    pd_["cdrsb"] = cdrsb
                    st.session_state.patient_step = 3
                    st.session_state.patient_data = pd_; st.rerun()

    # ── STEP 3: Mood Questions (GDS) ──────────────────────────────────────────
    elif step == 3:
        st.markdown("## 💬 Mood & Wellbeing")
        st.markdown("*Answer Yes or No for each question about how you have felt "
                    "**over the past week**.*")
        gds_score = 0
        with st.form("gds_form"):
            for i, (q, bad_ans) in enumerate(GDS_QUESTIONS):
                resp = st.radio(f"**{i+1}. {q}**", ["Yes", "No"],
                                horizontal=True, key=f"gds_{i}", index=1)
                if resp.lower() == bad_ans:
                    gds_score += 1
            c1, c2 = st.columns(2)
            with c1:
                if st.form_submit_button("← Back"):
                    st.session_state.patient_step = 2; st.rerun()
            with c2:
                if st.form_submit_button("Continue →", use_container_width=True):
                    pd_["gds"] = gds_score
                    st.session_state.patient_step = 4
                    st.session_state.patient_data = pd_; st.rerun()

    # ── STEP 4: Digit Span Test ───────────────────────────────────────────────
    elif step == 4:
        st.markdown("## 🔢 Number Memory Test")

        if "digit_sequences" not in st.session_state:
            base_lengths = [3, 4, 5, 6, 7, 8, 9]
            seqs = []
            for length in base_lengths:
                seq = random.sample(range(1, 10), min(length, 9))
                while len(seq) < length:
                    seq.append(random.randint(1, 9))
                seqs.append(seq)
            st.session_state.digit_sequences = seqs
            st.session_state.digit_level     = 0
            st.session_state.digit_correct   = 0
            st.session_state.digit_done      = False
            st.session_state.digit_phase     = "show"

        level = st.session_state.digit_level
        done  = st.session_state.digit_done
        seqs  = st.session_state.digit_sequences
        phase = st.session_state.digit_phase

        if done:
            raw_score        = st.session_state.digit_correct
            digit_span_score = min(raw_score * 2.5, 28)
            pd_["digit_span"]       = digit_span_score
            st.session_state.patient_data = pd_

            st.markdown(f"""<div class='pat-card' style='text-align:center'>
            <div style='font-size:2.5rem'>🎯</div>
            <h3>Number Test Complete!</h3>
            <p>You correctly recalled up to <b>{raw_score} digits</b>.</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🧲 Brain Scan Upload *(Optional)*")
            mri_uploaded = st.file_uploader("Upload brain scan (optional)",
                                             type=["nii", "gz"])
            if mri_uploaded:
                with st.spinner("Analysing brain scan…"):
                    try:
                        import nibabel as nib, torch
                        from PIL import Image
                        import torchvision.transforms as T
                        tmp = os.path.join(os.environ.get("TEMP", "C:\\Temp"),
                                           mri_uploaded.name)
                        with open(tmp, "wb") as f:
                            f.write(mri_uploaded.read())
                        d = np.array(nib.load(tmp).get_fdata(), dtype=np.float32)
                        if d.ndim == 4: d = d[..., 0]
                        nz = d.shape[2]; mid = int(nz * .5); sls = []
                        for off in [-1, 0, 1]:
                            sl = d[:, :, np.clip(mid + off, 0, nz - 1)]
                            lo, hi = np.percentile(sl, [1, 99])
                            sl = np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)
                            sls.append((sl * 255).astype(np.uint8))
                        if cnn_model:
                            rgb = np.stack(sls, 2)
                            pil = Image.fromarray(rgb, "RGB")
                            t   = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                             T.Normalize([.485,.456,.406],[.229,.224,.225])])
                            with torch.no_grad():
                                mp = float(torch.softmax(
                                    cnn_model(t(pil).unsqueeze(0)), 1)[0, 1])
                            st.session_state.mri_prob = mp
                            os.remove(tmp)
                            st.success(f"✅ Brain scan analysed. MRI risk score: {int(mp*100)}%")
                        else:
                            st.warning("CNN model not loaded — MRI skipped.")
                    except Exception as e:
                        st.warning(f"Could not process scan: {e}")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("← Redo Number Test"):
                    for k in ["digit_sequences", "digit_level", "digit_correct",
                              "digit_done", "digit_phase"]:
                        st.session_state.pop(k, None)
                    st.rerun()
            with c2:
                if st.button("Continue →", use_container_width=True):
                    st.session_state.patient_step = 5
                    st.session_state.patient_data = pd_; st.rerun()

        elif level >= len(seqs):
            st.session_state.digit_done = True; st.rerun()

        elif phase == "show":
            seq     = seqs[level]
            seq_js  = json.dumps([str(d) for d in seq])
            seq_str = "  ".join(str(d) for d in seq)

            st.markdown(f"*A sequence of **{len(seq)} numbers** will appear for "
                        "**5 seconds**, then disappear. Remember the order!*")
            st.markdown(f"**Sequence {level + 1} of {len(seqs)}:**")

            components.html(f"""
            <div id="phase-show" style="font-family:sans-serif">
              <div id="digit-display"
                style="font-family:monospace;font-size:3.5rem;font-weight:800;
                       letter-spacing:18px;text-align:center;color:#1e293b;
                       padding:28px;background:#f8fafc;border-radius:20px;
                       border:3px solid #e2e8f0;margin:12px 0">
                {seq_str}
              </div>
              <button id="listenBtn" onclick="readDigits()"
                style="background:#1e3a5f;color:white;border:none;border-radius:10px;
                       padding:10px 22px;font-size:0.92rem;font-weight:700;cursor:pointer">
                🔊 Read digits aloud</button>
              <div id="countdown-wrap" style="margin-top:16px;display:none">
                <div style="font-size:1rem;color:#7c3aed;font-weight:700">
                  Hiding in <span id="cdnum">5</span>s…</div>
                <div style="background:#ede9fe;border-radius:999px;height:8px;margin-top:6px">
                  <div id="cdbar" style="background:linear-gradient(90deg,#7c3aed,#db2777);
                       border-radius:999px;height:8px;width:100%;
                       transition:width 1s linear"></div>
                </div>
              </div>
            </div>
            <script>
            const digits = {seq_js}; let started = false;
            function readDigits() {{
              window.speechSynthesis.cancel();
              document.getElementById('listenBtn').disabled = true;
              let i = 0;
              function next() {{
                if(i < digits.length) {{
                  const u = new SpeechSynthesisUtterance(digits[i]);
                  u.rate = 0.7; u.onend = () => {{ i++; setTimeout(next, 400); }};
                  window.speechSynthesis.speak(u);
                }} else {{ if(!started) {{ started=true; startHideCountdown(); }} }}
              }}
              next();
            }}
            function startHideCountdown() {{
              document.getElementById('countdown-wrap').style.display='block';
              let t = 5;
              const iv = setInterval(() => {{
                t--; document.getElementById('cdnum').innerText = t;
                document.getElementById('cdbar').style.width = (t/5*100)+'%';
                if(t <= 0) {{
                  clearInterval(iv);
                  document.getElementById('digit-display').style.visibility='hidden';
                  document.getElementById('digit-display').innerText = '';
                  document.getElementById('countdown-wrap').innerHTML =
                    '<div style="color:#10b981;font-weight:700">✅ Numbers hidden! Type your answer below.</div>';
                }}
              }}, 1000);
            }}
            setTimeout(() => {{ if(!started) {{ started=true; startHideCountdown(); }} }}, 5000);
            </script>""", height=200)

            st.info("⏱ Numbers auto-hide after 5 seconds.")
            if st.button("Numbers are hidden — I'm ready to type →",
                          use_container_width=True, key=f"ready_{level}"):
                st.session_state.digit_phase = "type"; st.rerun()

        else:
            seq    = seqs[level]
            st.markdown(f"**Type the {len(seq)} numbers you just saw (no spaces):**")
            answer = st.text_input("Your answer:", key=f"ans_{level}", placeholder="")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("← See sequence again", key=f"back_{level}"):
                    st.session_state.digit_phase = "show"; st.rerun()
            with c2:
                if st.button("Submit →", use_container_width=True, key=f"sub_{level}"):
                    correct = "".join(str(d) for d in seq)
                    if answer.strip().replace(" ", "") == correct:
                        st.success("✅ Correct!")
                        st.session_state.digit_correct = level + 1
                        st.session_state.digit_level   = level + 1
                        st.session_state.digit_phase   = "show"
                    else:
                        st.error(f"❌ Correct answer was: **{correct}**")
                        st.session_state.digit_done = True
                    st.rerun()

    # ── STEP 5: Delayed Word Recall ───────────────────────────────────────────
    elif step == 5:
        st.markdown("## 🕐 Do You Remember the Words?")
        words_shown = st.session_state.get("words_shown", WORD_LIST)
        st.markdown("*Earlier, we showed you 15 words. Write down every word you can remember.*")
        st.markdown('<div class="pat-card">', unsafe_allow_html=True)
        delayed_input = st.text_area("Words you remember:", height=100,
                                      key="del_recall", placeholder="")
        show_hint = st.checkbox("I give up — show me the original list")
        if show_hint:
            word_html = "".join([f'<span class="word-box">{w.upper()}</span>'
                                  for w in words_shown])
            st.markdown(f'<div style="margin:10px 0">{word_html}</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Back"):
                st.session_state.patient_step = 4; st.rerun()
        with c2:
            if st.button("See My Results →", use_container_width=True):
                recalled_del = [w.strip().lower()
                                for w in delayed_input.replace(",", " ").split()
                                if w.strip()]
                words_set   = set(w.lower() for w in words_shown)
                correct_del = [w for w in recalled_del if w in words_set]
                correct_imm = pd_.get("immediate_recall", [])
                pd_["rv_del"]       = float(len(correct_del))
                pd_["rv_fo"]        = float(max(len(correct_imm) - len(correct_del), 0))
                pd_["correct_del"]  = correct_del
                st.session_state.patient_step = 6
                st.session_state.patient_data = pd_; st.rerun()

    # ── STEP 6: Results ───────────────────────────────────────────────────────
    elif step == 6:
        st.markdown("## 📊 Your Results")

        rv_del   = float(pd_.get("rv_del",    7))
        rv_fo    = float(pd_.get("rv_fo",     4))
        rv_imm   = float(pd_.get("rv_imm",   35))
        ds_score = float(pd_.get("digit_span",14))
        faq      = float(pd_.get("faq",       5))
        gds      = float(pd_.get("gds",       2))
        cdr_g    = float(pd_.get("cdr_g",    0.5))
        cdrsb    = float(pd_.get("cdrsb",    1.5))
        edu      = float(pd_.get("edu",      12))
        sex_val  = 0.0 if pd_.get("sex", "Male") in ["Male"] else 1.0

        mmse=26.0; moca=23.0; adas13=18.0; adas11=13.0; trails=120.0

        vals = {
            "PTGENDER":    sex_val,   "PTEDUCAT":  edu,
            "MMSE_BL":     mmse,      "MOCA_BL":   moca,
            "ADAS11_BL":   adas11,    "ADAS13_BL": adas13,
            "FAQ_BL":      faq,       "GDS_BL":    gds,
            "CDR_GLOBAL_BL": cdr_g,   "CDRSB_BL":  cdrsb,
            "RAVLT_forgetting": rv_fo, "RAVLT_immediate": rv_imm,
            "RAVLT_delayed":    rv_del, "DigitSpan": ds_score,
            "TrailsB":     trails,
        }
        vals = compute_composites(vals)

        if not primary:
            st.error("AI model not loaded."); st.stop()

        X    = np.array([[vals[f] for f in ALL_FEATURES]])
        prob = float(primary.predict_proba(X)[0, 1])

        mri_prob = st.session_state.get("mri_prob", None)
        if mri_prob is not None:
            prob = 0.75 * prob + 0.25 * mri_prob

        # Patient mode uses balanced threshold by default
        risk = risk_info(prob, threshold=0.35)
        st.session_state.risk = risk
        st.session_state.vals = vals

        # Summary metrics
        correct_del = pd_.get("correct_del", [])
        correct_imm = pd_.get("immediate_recall", [])
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Words recalled (immediate)", f"{len(correct_imm)} / 15")
        sc2.metric("Words recalled (after delay)", f"{len(correct_del)} / 15",
                   delta=str(len(correct_del) - len(correct_imm)) if correct_imm else None)
        sc3.metric("Digit Span", f"{int(ds_score)}")
        sc4.metric("Daily activities difficulty", f"{int(faq)} / 30")

        st.markdown("---")
        mri_note = (
            "<div style='margin-top:10px;font-size:0.82rem;color:#64748b'>"
            "Combined: clinical scores (75%) + brain scan (25%)</div>"
            if mri_prob else ""
        )
        st.markdown(
            f"""<div class="{risk['cls']}">
            <div style='font-size:3.5rem;margin-bottom:8px'>{risk['emoji']}</div>
            <div style='font-family:IBM Plex Mono,monospace;font-size:4rem;
                 font-weight:800;line-height:1;color:{risk["color"]}'>{risk['pct']}%</div>
            <div style='font-size:1.5rem;font-weight:800;color:{risk["color"]};
                 margin:10px 0'>{risk['headline']}</div>
            <div style='font-size:1rem;color:#374151;max-width:600px;
                 margin:0 auto;line-height:1.8'>{risk['plain']}</div>
            {mri_note}
            </div>""",
            unsafe_allow_html=True,
        )

        # Plain-language SHAP
        st.markdown("---")
        st.markdown("### 🔍 What is affecting the result most?")
        PLAIN_EXPLANATIONS = {
            "ADAS13_BL":          ("Thinking errors score",          "higher = more concern", True),
            "RAVLT_delayed":      ("Words remembered after a delay", "lower = more concern",  False),
            "DigitSpan":          ("Number memory (digit span)",     "lower = more concern",  False),
            "CDRSB_BL":           ("Daily function difficulty",      "higher = more concern", True),
            "FAQ_BL":             ("Daily activities difficulty",    "higher = more concern", True),
            "MMSE_BL":            ("Short memory test score",        "lower = more concern",  False),
            "RAVLT_immediate":    ("Words recalled in total",        "lower = more concern",  False),
            "RAVLT_forgetting":   ("Words forgotten over time",      "higher = more concern", True),
            "TrailsB":            ("Mental flexibility speed",       "slower = more concern", True),
            "MOCA_BL":            ("Overall thinking test",          "lower = more concern",  False),
            "CDR_GLOBAL_BL":      ("Overall memory rating",          "higher = more concern", True),
            "MMSE_FAQ_composite": ("Memory vs daily function gap",   "lower = more concern",  False),
            "ADAS_MMSE_gap":      ("Thinking difficulty index",      "higher = more concern", True),
            "RAVLT_forget_rate":  ("Rate of forgetting",             "higher = more concern", True),
            "GDS_BL":             ("Mood / depression signs",        "higher = more concern", True),
            "ADAS11_BL":          ("Thinking test errors",           "higher = more concern", True),
            "PTEDUCAT":           ("Years of education",             "fewer years = some risk", False),
            "PTGENDER":           ("Sex",                            "minor factor",           False),
        }

        try:
            if hasattr(primary, "steps"):
                from sklearn.pipeline import Pipeline as SKPipeline
                xgb_model = primary.steps[-1][1]
                if len(primary.steps) > 1:
                    pre = SKPipeline(primary.steps[:-1])
                    try:    X_shap = pre.transform(X)
                    except: X_shap = X
                else:
                    X_shap = X
            else:
                xgb_model = primary
                X_shap    = X

            exp      = shap.TreeExplainer(xgb_model)
            sv       = exp.shap_values(X_shap)[0]
            top_idx  = np.argsort(np.abs(sv))[-8:][::-1]
            factors_up, factors_down = [], []

            for i in top_idx:
                feat     = ALL_FEATURES[i]
                shap_val = sv[i]
                label, hint, _ = PLAIN_EXPLANATIONS.get(feat, (feat, "", True))
                if shap_val > 0.05:
                    factors_up.append((label, hint, abs(shap_val)))
                elif shap_val < -0.05:
                    factors_down.append((label, hint, abs(shap_val)))

            if factors_up:
                st.markdown("#### 🔴 Factors raising your risk")
                for label, hint, magnitude in sorted(factors_up, key=lambda x: -x[2]):
                    bar_w = min(int(magnitude * 300), 100)
                    st.markdown(f"""<div style='background:white;border-left:4px solid #ef4444;
                    border-radius:10px;padding:14px 18px;margin:6px 0;box-shadow:0 1px 8px #0001'>
                    <div style='font-weight:700;font-size:0.95rem;color:#1e293b'>{label}</div>
                    <div style='font-size:0.82rem;color:#64748b;margin:2px 0'>{hint}</div>
                    <div style='background:#fee2e2;border-radius:999px;height:5px;margin-top:8px'>
                      <div style='background:#ef4444;border-radius:999px;height:5px;
                           width:{bar_w}%'></div></div></div>""",
                    unsafe_allow_html=True)

            if factors_down:
                st.markdown("#### 🟢 Factors working in your favour")
                for label, hint, magnitude in sorted(factors_down, key=lambda x: -x[2]):
                    bar_w = min(int(magnitude * 300), 100)
                    st.markdown(f"""<div style='background:white;border-left:4px solid #10b981;
                    border-radius:10px;padding:14px 18px;margin:6px 0;box-shadow:0 1px 8px #0001'>
                    <div style='font-weight:700;font-size:0.95rem;color:#1e293b'>{label}</div>
                    <div style='font-size:0.82rem;color:#64748b;margin:2px 0'>{hint}</div>
                    <div style='background:#d1fae5;border-radius:999px;height:5px;margin-top:8px'>
                      <div style='background:#10b981;border-radius:999px;height:5px;
                           width:{bar_w}%'></div></div></div>""",
                    unsafe_allow_html=True)

        except Exception as e:
            st.info(f"Could not generate factor breakdown: {e}")

        # Prevention
        st.markdown("---")
        st.markdown("### 🛡️ What to do next")
        for title, desc, cls in PREVENTION.get(risk["level"], PREVENTION["MODERATE"])[:5]:
            st.markdown(f"""<div class="{cls}">
            <div style='font-weight:700;font-size:0.95rem;margin-bottom:5px'>{title}</div>
            <div style='font-size:0.87rem;color:#374151;line-height:1.6'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Redo Assessment"):
                st.session_state.patient_step = 0
                st.session_state.patient_data = {}
                for k in ["words_shown", "word_phase", "digit_sequences", "digit_level",
                          "digit_correct", "digit_done", "digit_phase", "mri_prob"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with c2:
            st.markdown("""<div style='background:#eff6ff;border:1px solid #bfdbfe;
            border-radius:12px;padding:14px 18px;font-size:0.87rem;color:#1d4ed8'>
            <b>💡 For higher accuracy:</b> Ask your doctor for clinical test scores
            and enter them in <b>Doctor Mode</b>.
            </div>""", unsafe_allow_html=True)