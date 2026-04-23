# app/app.py — Final Production Version
import os
import sys
import numpy as np
from PIL import Image
import streamlit as st

# ─── Path Setup ───────────────────────────────────────────────────────────────
app_dir     = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(app_dir)
MODEL_PATH  = os.path.join(project_dir, "model", "grey_water_model.h5")
IMG_SIZE    = (224, 224)
CLASS_NAMES = ['high', 'low', 'medium']

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Grey Water Classifier",
    page_icon  = "💧",
    layout     = "centered"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .title {
        text-align: center;
        color: #1a73e8;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .low-box    { background: #eafaf1; border-left: 5px solid #27ae60; }
    .medium-box { background: #fef9e7; border-left: 5px solid #f39c12; }
    .high-box   { background: #fdedec; border-left: 5px solid #e74c3c; }
    .result-label {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .low-text    { color: #27ae60; }
    .medium-text { color: #f39c12; }
    .high-text   { color: #e74c3c; }
    .desc { color: #555; font-size: 0.95rem; }
    .footer {
        text-align: center;
        color: #bbb;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Class Config ─────────────────────────────────────────────────────────────
CLASS_CONFIG = {
    'low': {
        'label'      : 'LOW TURBIDITY',
        'emoji'      : '🟢',
        'box_class'  : 'low-box',
        'text_class' : 'low-text',
        'description': 'Water appears clear. Safe for grey water reuse.',
        'advice'     : 'Suitable for toilet flushing, garden irrigation, or cleaning.',
        'color'      : '#27ae60'
    },
    'medium': {
        'label'      : 'MEDIUM TURBIDITY',
        'emoji'      : '🟡',
        'box_class'  : 'medium-box',
        'text_class' : 'medium-text',
        'description': 'Water appears cloudy or milky. Limited reuse potential.',
        'advice'     : 'Suitable only for subsurface irrigation. Avoid direct contact.',
        'color'      : '#f39c12'
    },
    'high': {
        'label'      : 'HIGH TURBIDITY',
        'emoji'      : '🔴',
        'box_class'  : 'high-box',
        'text_class' : 'high-text',
        'description': 'Water is very murky. High contamination risk.',
        'advice'     : 'Requires treatment before any reuse. Do not use directly.',
        'color'      : '#e74c3c'
    }
}

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
    return tf.keras.models.load_model(MODEL_PATH)

# ─── Predict ──────────────────────────────────────────────────────────────────
def predict(image, model):
    img  = image.convert('RGB').resize(IMG_SIZE)
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)
    idx  = np.argmax(pred[0])
    return CLASS_NAMES[idx], pred[0]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="title">💧 Grey Water Quality Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered turbidity detection using MobileNetV2</p>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system uses **Transfer Learning** with MobileNetV2 to classify grey water into:

    - 🟢 **Low** — Clear water
    - 🟡 **Medium** — Cloudy water
    - 🔴 **High** — Murky water

    ---
    **Model:** MobileNetV2
    **Input:** 224×224 RGB image
    **Classes:** 3
    **Framework:** TensorFlow
    """)

    st.header("📋 Tips")
    st.markdown("""
    - Use clear, well-lit photos
    - Keep the glass/container in frame
    - Avoid blurry images
    - Plain background works best
    """)

# ─── Load Model ───────────────────────────────────────────────────────────────
with st.spinner("Loading AI model..."):
    model = load_model()
st.success("Model ready!", icon="✅")

# ─── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a water image",
    type=["jpg", "jpeg", "png"],
    help="Supported: JPG, JPEG, PNG"
)

# ─── Results ──────────────────────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        # Image metadata
        st.caption(f"File: {uploaded_file.name}")
        st.caption(f"Size: {image.size[0]}×{image.size[1]} px")

    with col2:
        st.subheader("Analysis Result")

        with st.spinner("Analyzing..."):
            predicted_class, probabilities = predict(image, model)

        cfg = CLASS_CONFIG[predicted_class]

        # Result box
        st.markdown(f"""
        <div class="result-box {cfg['box_class']}">
            <div class="result-label {cfg['text_class']}">
                {cfg['emoji']} {cfg['label']}
            </div>
            <div class="desc">{cfg['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence scores
        st.markdown("**Confidence Scores:**")
        for i, cls in enumerate(CLASS_NAMES):
            c   = CLASS_CONFIG[cls]
            pct = float(probabilities[i])
            st.markdown(f"{c['emoji']} **{cls.capitalize()}** — {pct*100:.1f}%")
            st.progress(pct)

        # Recommendation
        st.markdown("---")
        st.markdown(f"**Recommendation:**")
        st.info(cfg['advice'])

        # Download result as text
        result_text = f"""
Grey Water Quality Classification Report
=========================================
File      : {uploaded_file.name}
Prediction: {cfg['label']}
Confidence: {max(probabilities)*100:.1f}%

Scores:
  High   : {probabilities[0]*100:.1f}%
  Low    : {probabilities[1]*100:.1f}%
  Medium : {probabilities[2]*100:.1f}%

Recommendation: {cfg['advice']}
        """
        st.download_button(
            label    = "Download Report",
            data     = result_text,
            file_name= f"water_report_{uploaded_file.name}.txt",
            mime     = "text/plain"
        )

# ─── Instructions when no image uploaded ──────────────────────────────────────
else:
    st.markdown("""
    ### How to use:
    1. Click **Browse files** above
    2. Select a water image (JPG or PNG)
    3. AI will instantly classify turbidity

    ### Turbidity Classes:
    | Level | Visual | Recommendation |
    |-------|--------|----------------|
    | 🟢 Low | Clear water | Safe for reuse |
    | 🟡 Medium | Cloudy/milky | Limited reuse |
    | 🔴 High | Murky/opaque | Needs treatment |
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p class="footer">Grey Water Classification System • MobileNetV2 • TensorFlow</p>',
    unsafe_allow_html=True
)