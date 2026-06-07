"""
PsikisAI: Mental Health Condition Detection from Social Media Texts
-------------------------------------------------------------------
Streamlit application (localhost) using the fine-tuned RoBERTa model (`roberta_best`).

Pipeline notes (matched to the training notebook):
  - The transformer was trained on RAW text (the clean_text() function was only
    used for the TF-IDF baselines), so user input is fed directly to the tokenizer.
  - Tokenizer: truncation=True, max_length=128 (MAX_LEN_TRANS in the notebook).
  - Prediction: softmax(logits) -> argmax.
  - 6 classes (read from the model config): Anxiety, Bipolar, Depression,
    Normal, Stress, Suicidal.

Run:
  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# Configuration
MODEL_DIR = os.environ.get("PSIKISAI_MODEL_DIR", "roberta_best")
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="PsikisAI Mental Health Text Insight",
    page_icon="🪻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Per-class presentation metadata.
CLASS_META = {
    "Anxiety":    {"emoji": "🌫️", "color": "#E0A24A",
                   "desc": "Language patterns suggesting worry, unease, or racing thoughts."},
    "Bipolar":    {"emoji": "🌗", "color": "#B6789F",
                   "desc": "Patterns hinting at noticeably sharp swings between emotional highs and lows."},
    "Depression": {"emoji": "🌧️", "color": "#7C93B0",
                   "desc": "Language leaning toward deep sadness, emptiness, or loss of interest."},
    "Normal":     {"emoji": "🌿", "color": "#7FA98A",
                   "desc": "No prominent indication of a specific condition detected in this text."},
    "Stress":     {"emoji": "🔥", "color": "#E0785A",
                   "desc": "Patterns suggesting pressure, feeling overwhelmed, or a mounting workload."},
    "Suicidal":   {"emoji": "🕊️", "color": "#C75D5D",
                   "desc": "Text contains signals related to hopelessness or thoughts of harming oneself."},
}

# Indonesian crisis-support resources (verified, 2025/2026).
SUPPORT_RESOURCES = [
    ("Healing119.id (Ministry of Health)", "Call <b>119 ext. 8</b> or chat at <b>www.healing119.id</b>",
     "Free, 24/7, anonymous national psychological first aid (Indonesia)."),
    ("JakCare (Jakarta residents)", "Call <b>0800 1500 119</b> or use the <b>JAKI</b> app",
     "24/7, handles emergencies including crisis situations."),
    ("Halo Kemenkes", "Call <b>1500-567</b>", "24/7 national health information line (Indonesia)."),
    ("Yayasan Pulih", "WhatsApp <b>+62 811 8436 633</b>", "Psychological counseling (individual, couples, family)."),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Nunito:wght@400;500;600;700;800&display=swap');

:root {
  --cream: #FBF5EE;
  --peach: #FDEDE3;
  --peach-deep: #F8DCC9;
  --ink: #4A3F3A;
  --ink-soft: #8A7B72;
  --coral: #E0785A;
  --coral-deep: #C75D45;
  --sage: #7FA98A;
}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] * {
  font-family: 'Nunito', sans-serif;
}
/* Keep Material icons (e.g. the sidebar arrow) rendering as glyphs, not raw text */
span[class*="material-symbols"], span[class*="material-icons"],
[data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] span,
[data-testid="collapsedControl"] span {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
               'Material Icons', 'Material Icons Round' !important;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 500px at 12% -5%, #FCEFE3 0%, rgba(252,239,227,0) 55%),
    radial-gradient(900px 500px at 100% 0%, #F7E4D6 0%, rgba(247,228,214,0) 50%),
    var(--cream);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1100px; }

h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--ink); letter-spacing: -0.01em; }

/* Hero */
.hero {
  background: linear-gradient(135deg, #FFFFFF 0%, #FFF6EF 100%);
  border: 1px solid #F2DECE;
  border-radius: 26px;
  padding: 30px 34px;
  box-shadow: 0 18px 40px -22px rgba(199,93,69,0.35);
  margin-bottom: 22px;
}
.hero-title { font-family:'Fraunces',serif; font-size: 2.5rem; font-weight: 600; margin: 0; color: var(--ink); }
.hero-title .accent { color: var(--coral); }
.hero-sub { color: var(--ink-soft); font-size: 1.04rem; margin-top: 8px; max-width: 720px; line-height: 1.5; }
.hero-pills { margin-top: 16px; }
.pill {
  display:inline-block; background: var(--peach); color: var(--coral-deep);
  border:1px solid var(--peach-deep); padding: 5px 14px; border-radius: 999px;
  font-size: 0.82rem; font-weight: 700; margin-right: 8px; margin-bottom: 6px;
}

/* Cards */
.card {
  background: #FFFFFF; border: 1px solid #F2DECE; border-radius: 22px;
  padding: 22px 24px; box-shadow: 0 14px 34px -26px rgba(74,63,58,0.45); margin-bottom: 18px;
}
.section-kicker { font-size: 0.78rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--coral); margin-bottom: 4px; }

/* Result hero card */
.result-card {
  display:flex; align-items:center; gap: 22px;
  background: linear-gradient(135deg, #FFFFFF 0%, #FFF4EC 100%);
  border: 1px solid #F2DECE; border-left: 8px solid var(--accent);
  border-radius: 22px; padding: 24px 26px; margin-bottom: 16px;
  box-shadow: 0 16px 36px -26px rgba(74,63,58,0.4);
}
.result-emoji { font-size: 3rem; line-height:1; }
.result-text { flex: 1; }
.result-kicker { font-size: 0.76rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ink-soft); }
.result-name { font-family:'Fraunces',serif; font-size: 1.9rem; font-weight: 600; color: var(--ink); margin: 2px 0 4px; }
.result-desc { color: var(--ink-soft); font-size: 0.96rem; line-height: 1.45; }
.result-conf { text-align:center; min-width: 92px; }
.conf-num { font-family:'Fraunces',serif; font-size: 2rem; font-weight: 700; color: var(--accent); line-height: 1; }
.conf-cap { font-size: 0.74rem; color: var(--ink-soft); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }

/* Probability bars */
.prob-list { margin-top: 6px; }
.prob-row { display:flex; align-items:center; gap: 12px; padding: 7px 0; }
.prob-row.pred .prob-label { font-weight: 800; color: var(--ink); }
.prob-label { width: 168px; font-size: 0.95rem; color: #6b5d55; font-weight: 600; }
.prob-label .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }
.prob-track { flex:1; height: 14px; background: #F3E7DC; border-radius: 999px; overflow: hidden; }
.prob-fill { height:100%; border-radius:999px; transition: width .6s ease; }
.prob-pct { width: 56px; text-align:right; font-weight: 700; color: var(--ink); font-size: 0.92rem; }

/* LIME chips */
.lime-wrap { display:flex; flex-wrap:wrap; gap: 8px; margin-top: 6px; }
.lime-chip { border-radius: 12px; padding: 6px 12px; font-weight: 700; font-size: 0.92rem;
  color: #4a3f3a; border: 1px solid rgba(74,63,58,0.06); display:inline-flex; align-items:center; gap:8px; }
.lime-chip .lime-w { font-size: 0.74rem; font-weight: 700; opacity: 0.6; }
.lime-legend { font-size: 0.84rem; color: var(--ink-soft); margin-top: 10px; }
.lime-legend b.pos { color: var(--coral-deep); }
.lime-legend b.neg { color: #5f7d9a; }

/* Support box */
.support-box {
  background: linear-gradient(135deg, #FFF1EC 0%, #FCE6DC 100%);
  border: 1px solid #F4CBB8; border-radius: 22px; padding: 22px 24px; margin-bottom: 18px;
}
.support-title { font-family:'Fraunces',serif; font-size: 1.25rem; color: var(--coral-deep); margin: 0 0 4px; }
.support-lead { color: #7a5a4c; font-size: 0.96rem; line-height: 1.5; margin-bottom: 14px; }
.support-item { background: #FFFFFF; border:1px solid #F4D6C7; border-radius: 14px;
  padding: 12px 16px; margin-bottom: 10px; }
.support-item .si-name { font-weight: 800; color: var(--ink); }
.support-item .si-contact { color: var(--coral-deep); font-size: 0.96rem; margin: 2px 0; }
.support-item .si-note { color: var(--ink-soft); font-size: 0.84rem; }

/* Disclaimer */
.disclaimer { background: #FFFDF9; border: 1px dashed #E7C9B6; border-radius: 16px;
  padding: 14px 18px; color: var(--ink-soft); font-size: 0.86rem; line-height: 1.5; }

/* Buttons */
.stButton > button {
  background: var(--coral); color: #fff; border: none; border-radius: 14px;
  padding: 0.55rem 1.4rem; font-weight: 800; font-family: 'Nunito', sans-serif;
  box-shadow: 0 10px 22px -12px rgba(199,93,69,0.7); transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 14px 26px -12px rgba(199,93,69,0.8);
  background: var(--coral-deep); color:#fff; }

/* Inputs */
[data-testid="stTextArea"] textarea {
  border-radius: 16px !important; border: 1.5px solid #EAD3C2 !important;
  background: #FFFDFB !important; font-size: 1rem !important; color: var(--ink) !important;
}
[data-testid="stTextArea"] textarea:focus { border-color: var(--coral) !important; box-shadow: 0 0 0 3px rgba(224,120,90,0.15) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #FFF6EF 0%, #FBEADD 100%); border-right: 1px solid #F2DECE; }
[data-testid="stSidebar"] .stMarkdown { color: var(--ink); }

/* Tabs */
[data-baseweb="tab-list"] { gap: 6px; }
[data-baseweb="tab"] { background: #FFF4EC; border-radius: 12px 12px 0 0; font-weight: 700; }
[aria-selected="true"][data-baseweb="tab"] { background: #FFFFFF; color: var(--coral-deep); }

footer, #MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Model loading (cached once)
@st.cache_resource(show_spinner="Loading the RoBERTa model (one time only)...")
def load_model(model_dir: str):
    if not os.path.isdir(model_dir):
        return None, None, None
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE).eval()
    n = model.config.num_labels
    class_names = [model.config.id2label[i] for i in range(n)]
    return tokenizer, model, class_names


def predict_proba(texts):
    """Return softmax probabilities. Raw text -> tokenizer (matches training)."""
    if isinstance(texts, str):
        texts = [texts]
    enc = TOKENIZER(list(texts), truncation=True, padding=True,
                    max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = MODEL(**enc).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()


# Rendering helpers
def render_result_card(name: str, conf: float) -> str:
    m = CLASS_META[name]
    return f"""
    <div class="result-card" style="--accent:{m['color']}">
      <div class="result-emoji">{m['emoji']}</div>
      <div class="result-text">
        <div class="result-kicker">Top prediction</div>
        <div class="result-name">{name}</div>
        <div class="result-desc">{m['desc']}</div>
      </div>
      <div class="result-conf">
        <div class="conf-num" style="color:{m['color']}">{conf*100:.0f}%</div>
        <div class="conf-cap">confidence</div>
      </div>
    </div>"""


def render_prob_bars(probs, class_names, pred_idx) -> str:
    html = '<div class="prob-list">'
    for i in np.argsort(probs)[::-1]:
        name = class_names[i]
        m = CLASS_META.get(name, {"emoji": "•", "color": "#E0785A"})
        pct = probs[i] * 100
        pred_cls = "pred" if i == pred_idx else ""
        html += f"""
        <div class="prob-row {pred_cls}">
          <div class="prob-label"><span class="dot" style="background:{m['color']}"></span>{m['emoji']} {name}</div>
          <div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%;background:{m['color']}"></div></div>
          <div class="prob-pct">{pct:.1f}%</div>
        </div>"""
    return html + "</div>"


def render_lime(word_weights) -> str:
    if not word_weights:
        return "<p style='color:#8A7B72'>No words were influential enough to display.</p>"
    max_w = max(abs(w) for _, w in word_weights) or 1.0
    html = '<div class="lime-wrap">'
    for word, weight in word_weights:
        intensity = min(abs(weight) / max_w, 1.0)
        if weight >= 0:
            bg = f"rgba(224,120,90,{0.16 + 0.55*intensity:.2f})"
        else:
            bg = f"rgba(124,147,176,{0.14 + 0.45*intensity:.2f})"
        html += f'<span class="lime-chip" style="background:{bg}">{word}<span class="lime-w">{weight:+.2f}</span></span>'
    html += "</div>"
    html += ('<div class="lime-legend">'
             '<b class="pos">Warm color</b> = word pushes TOWARD this prediction &middot; '
             '<b class="neg">Cool color</b> = word pushes AWAY from this prediction.</div>')
    return html


def render_support_box(predicted: str) -> str:
    if predicted == "Suicidal":
        lead = ("This text carries some heavy signals. If it reflects how you or someone close to you "
                "feels right now, please know you are not alone. Someone is ready to listen, any time.")
    else:  # Depression
        lead = ("If this text reflects how you have been feeling lately, talking to someone who is ready "
                "to listen can help. Here are a few safe places to start.")
    items = ""
    for name, contact, note in SUPPORT_RESOURCES:
        items += (f'<div class="support-item"><div class="si-name">{name}</div>'
                  f'<div class="si-contact">{contact}</div>'
                  f'<div class="si-note">{note}</div></div>')
    return (f'<div class="support-box"><div class="support-title">💛 You are not alone</div>'
            f'<div class="support-lead">{lead}</div>{items}</div>')


DISCLAIMER_HTML = (
    '<div class="disclaimer"><b>Important note.</b> PsikisAI is an academic project and '
    '<b>not a diagnostic tool</b>. Its results only detect language patterns in text, not a '
    "person's clinical condition. For anything related to mental health, always consult a "
    'psychologist, psychiatrist, or qualified professional.</div>'
)

# Load model
TOKENIZER, MODEL, CLASS_NAMES = load_model(MODEL_DIR)

# Sidebar
with st.sidebar:
    st.markdown("### 🪻 PsikisAI")
    st.markdown(
        "Detects indications of mental health conditions from social media text, "
        "powered by a fine-tuned **RoBERTa** model."
    )
    st.markdown("---")
    st.markdown("**Model**")
    if MODEL is not None:
        st.markdown(
            f"- Architecture: `roberta-base`\n"
            f"- Number of classes: **{len(CLASS_NAMES)}**\n"
            f"- Device: `{DEVICE}`"
        )
        st.caption("Classes: " + ", ".join(CLASS_NAMES))
    else:
        st.error(f"Model folder `{MODEL_DIR}` not found.")
    st.markdown("---")
    st.markdown("**Quick help (Indonesia)**")
    st.markdown(
        "- Healing119.id: **119 ext. 8**\n"
        "- JakCare (Jakarta): **0800 1500 119**\n"
        "- Yayasan Pulih: **+62 811 8436 633**"
    )
    st.caption("Free & confidential, available 24/7.")

# Hero
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">Psikis<span class="accent">AI</span></div>
      <div class="hero-sub">Paste a social media post, and PsikisAI reads its language patterns to
      estimate possible indications of a mental health condition, complete with a confidence level
      and an explanation of which words mattered most.</div>
      <div class="hero-pills">
        <span class="pill">RoBERTa fine-tuned</span>
        <span class="pill">6 classes</span>
        <span class="pill">LIME explanation</span>
        <span class="pill">Not a diagnostic tool</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if MODEL is None:
    st.error(
        f"The model could not be loaded. Make sure the **`{MODEL_DIR}`** folder "
        "(containing `config.json`, `model.safetensors`, and the tokenizer files) is "
        "in the same directory as `app.py`."
    )
    st.stop()

# Tabs
tab_single, tab_batch, tab_about = st.tabs(
    ["  ✍️  Single Text  ", "  📂  Batch Analysis  ", "  ℹ️  About  "]
)

# Tab 1: single text
with tab_single:
    EXAMPLES = {
        "Choose an example": "",
        "Example: depression": "I haven't been able to get out of bed for weeks, everything just feels pointless and grey.",
        "Example: anxious": "My chest feels tight and my heart won't stop pounding, I'm terrified something awful is about to happen.",
        "Example: stressed": "I'm so stressed and overwhelmed lately, the constant pressure at work is burning me out and I can't cope.",
        "Example: doing fine": "Had a great coffee with friends this morning, looking forward to the weekend trip!",
    }
    def _apply_example():
        sentence = EXAMPLES.get(st.session_state.get("example_select", ""), "")
        if sentence:
            st.session_state["user_text"] = sentence

    c1, c2 = st.columns([3, 1])
    with c2:
        st.selectbox("Try an example", list(EXAMPLES.keys()),
                     key="example_select", on_change=_apply_example,
                     label_visibility="collapsed")
    with c1:
        st.markdown('<div class="section-kicker">Text to analyze</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "Text", height=150, key="user_text",
        placeholder="Write or paste a post here... (English, matching the model's training data)",
        label_visibility="collapsed",
    )

    analyze = st.button("Analyze text", type="primary")

    if analyze:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                probs = predict_proba(user_text)[0]
            pred_idx = int(np.argmax(probs))
            st.session_state["result"] = {
                "text": user_text, "probs": probs.tolist(), "pred_idx": pred_idx,
            }
            st.session_state.pop("lime", None)  # reset old explanation

    if "result" in st.session_state:
        res = st.session_state["result"]
        probs = np.array(res["probs"])
        pred_idx = res["pred_idx"]
        pred_name = CLASS_NAMES[pred_idx]

        st.markdown(render_result_card(pred_name, probs[pred_idx]), unsafe_allow_html=True)

        if pred_name in ("Suicidal", "Depression"):
            st.markdown(render_support_box(pred_name), unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="section-kicker">Probability breakdown</div>',
                    unsafe_allow_html=True)
        st.markdown(render_prob_bars(probs, CLASS_NAMES, pred_idx), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="section-kicker">Why this prediction?</div>',
                    unsafe_allow_html=True)
        st.caption("LIME highlights the words that most influenced the model's decision. "
                   "This runs on CPU and takes roughly 20 to 60 seconds.")
        if st.button("Explain with LIME  ✨"):
            with st.spinner("Computing LIME explanation..."):
                explainer = LimeTextExplainer(class_names=CLASS_NAMES)
                exp = explainer.explain_instance(
                    res["text"], predict_proba,
                    num_features=12, num_samples=400, labels=[pred_idx],
                )
                st.session_state["lime"] = exp.as_list(label=pred_idx)
        if "lime" in st.session_state:
            st.markdown(render_lime(st.session_state["lime"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)

# Tab 2: batch
with tab_batch:
    st.markdown('<div class="section-kicker">Analyze many texts at once</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload a **CSV** file (then pick the text column) or a **TXT** file (one post per line). "
        "PsikisAI will predict every row and summarize the distribution."
    )
    up = st.file_uploader("Upload file", type=["csv", "txt"], label_visibility="collapsed")

    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df_in = pd.read_csv(up)
                text_col = st.selectbox("Column containing the text", df_in.columns.tolist())
                texts = df_in[text_col].fillna("").astype(str).tolist()
            else:
                raw = up.read().decode("utf-8", errors="ignore")
                texts = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                df_in = pd.DataFrame({"text": texts})
                text_col = "text"

            texts = [t for t in texts if t.strip()]
            st.caption(f"{len(texts)} rows ready to analyze.")

            if st.button("Run batch analysis", type="primary"):
                progress = st.progress(0.0, text="Predicting...")
                preds, confs = [], []
                BATCH = 16
                for start in range(0, len(texts), BATCH):
                    chunk = texts[start:start + BATCH]
                    p = predict_proba(chunk)
                    idx = p.argmax(axis=1)
                    preds.extend(CLASS_NAMES[i] for i in idx)
                    confs.extend(p.max(axis=1).tolist())
                    progress.progress(min((start + BATCH) / len(texts), 1.0), text="Predicting...")
                progress.empty()

                out = pd.DataFrame({
                    "text": texts,
                    "prediction": preds,
                    "confidence": [round(c, 4) for c in confs],
                })
                st.session_state["batch_out"] = out

            if "batch_out" in st.session_state:
                out = st.session_state["batch_out"]
                colA, colB = st.columns([1, 1])
                with colA:
                    st.markdown("**Prediction distribution**")
                    dist = out["prediction"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
                    st.bar_chart(dist)
                with colB:
                    st.markdown("**Summary**")
                    st.metric("Total texts", len(out))
                    top = out["prediction"].mode()
                    if len(top):
                        st.metric("Most common class", top.iloc[0])
                    st.metric("Average confidence", f"{out['confidence'].mean()*100:.1f}%")

                st.markdown("**Results per text**")
                st.dataframe(out, use_container_width=True, height=320)
                st.download_button(
                    "⬇️  Download results (CSV)",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="psikisai_results.csv", mime="text/csv",
                )
        except Exception as e:
            st.error(f"Failed to process file: {e}")

    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)

# Tab 3: about
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### About PsikisAI")
    st.markdown(
        "**PsikisAI** detects indications of mental health conditions from social media text and groups "
        "them into several categories. The primary model behind this app is **RoBERTa-base**, fine-tuned "
        "on the *Sentiment Analysis for Mental Health* dataset (Kaggle)."
    )
    st.markdown("#### Recognized classes")
    legend = ""
    for name in CLASS_NAMES:
        m = CLASS_META.get(name, {"emoji": "•", "color": "#E0785A", "desc": ""})
        legend += (f'<div style="display:flex;gap:12px;align-items:flex-start;margin:8px 0;">'
                   f'<span style="font-size:1.4rem">{m["emoji"]}</span>'
                   f'<div><b style="color:{m["color"]}">{name}</b>'
                   f'<div style="color:#8A7B72;font-size:0.9rem">{m["desc"]}</div></div></div>')
    st.markdown(legend, unsafe_allow_html=True)
    st.markdown(
        "#### How it works\n"
        "1. Raw text is tokenized directly (no cleaning), matching how the model was trained.\n"
        "2. The model computes a probability for each class (*softmax*).\n"
        "3. The class with the highest probability becomes the prediction.\n"
        "4. **LIME** explains which words most influenced the decision."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
