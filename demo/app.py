import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import streamlit as st
from transformers import AutoTokenizer

from src.training.config import ID2LABEL, TrainingConfig
from src.model.dual_head_xlmr import DualHeadConfig, DualHeadXLMRClassifier

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "ablation_no_norm_best.pt")

CLASS_COLORS = {
    "Not Hope": "#ff6b6b",
    "Generalized Hope": "#51cf66",
    "Realistic Hope": "#339af0",
    "Unrealistic Hope": "#fcc419",
}

CLASS_DESCRIPTIONS = {
    "Not Hope": "The text does not express hope.",
    "Generalized Hope": "The text expresses general optimism or support.",
    "Realistic Hope": "The text expresses hope grounded in achievable goals.",
    "Unrealistic Hope": "The text expresses hope that is unlikely to be fulfilled.",
}


@st.cache_resource
def load_model():
    cfg = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dual_cfg = DualHeadConfig(model_name=cfg.model_name, num_labels=cfg.num_labels)
    model = DualHeadXLMRClassifier(dual_cfg).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    return model, tokenizer, device


def predict(text: str, model, tokenizer, device, max_length: int = 128):
    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    pred_id = int(torch.argmax(logits, dim=-1).item())
    return pred_id, probs


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="HLSP Hope Speech Detection", layout="centered")
st.title("Hope Speech Detection in Roman Urdu")
st.caption("Hybrid Lexical-Semantic Pipeline (HLSP) · DualHead XLM-RoBERTa · NUST SEECS")

with st.spinner("Loading model..."):
    model, tokenizer, device = load_model()

st.success(f"Model loaded — running on **{'GPU' if device.type == 'cuda' else 'CPU'}**")

st.markdown("---")

examples = [
    "inshaallah sab theek ho jaye ga, himmat rakhein",
    "yeh mulk kabhi nahi bachega, sab barbaad hai",
    "kal exam hai aur main zaroor pass ho jaunga kyunki main ne khub parha hai",
    "kaash lottery lag jaye to sab mushkilat khatam",
]

st.markdown("**Try an example:**")
cols = st.columns(2)
selected = None
for i, ex in enumerate(examples):
    if cols[i % 2].button(f"Example {i+1}", key=f"ex{i}"):
        selected = ex

text_input = st.text_area(
    "Enter Roman Urdu text:",
    value=selected if selected else "",
    height=120,
    placeholder="e.g. inshaallah sab theek ho jaye ga...",
)

if st.button("Analyze", type="primary") and text_input.strip():
    pred_id, probs = predict(text_input.strip(), model, tokenizer, device)
    pred_label = ID2LABEL[pred_id]
    color = CLASS_COLORS[pred_label]

    st.markdown("---")
    st.markdown(f"### Prediction")
    st.markdown(
        f"<div style='background:{color}22; border-left:5px solid {color}; "
        f"padding:12px 16px; border-radius:6px; font-size:1.2em;'>"
        f"<b>{pred_label}</b> &nbsp;·&nbsp; {CLASS_DESCRIPTIONS[pred_label]}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Confidence Scores")
    for i, label in ID2LABEL.items():
        pct = probs[i] * 100
        bar_color = CLASS_COLORS[label]
        st.markdown(
            f"<div style='margin:4px 0'>"
            f"<span style='display:inline-block;width:160px'>{label}</span>"
            f"<div style='display:inline-block;width:{pct*3:.0f}px;height:14px;"
            f"background:{bar_color};border-radius:3px;vertical-align:middle'></div>"
            f" <span style='font-size:0.9em'>{pct:.1f}%</span></div>",
            unsafe_allow_html=True,
        )

elif st.button("Analyze", type="primary"):
    st.warning("Please enter some text first.")

st.markdown("---")
st.caption(
    "Model: DualHead XLM-RoBERTa (no normalizer) · "
    "Dev Macro F1: 0.5550 · Weighted F1: 0.6603 · "
    "Muhammad Sohaib Akhtar & Nosherwan Tahir · NUST SEECS 2026"
)
