"""
AgriVision AI - Crop Disease Detection System
Streamlit frontend for ResNet18-based plant disease classification.
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
import io

from model.model_loader import load_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from utils.preprocessing import transform
from data.disease_data import CLASS_NAMES, DISEASE_INFO

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AgriVision AI | Crop Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------
# PEAK UI - Premium AI Product Design
# ------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* Reset Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
button[data-testid="baseButton-secondary"] { visibility: hidden; }
div[data-testid="stProgress"] { display: none; }

/* Base */
.stApp {
    background: #0a0a0f !important;
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(120, 80, 255, 0.15), transparent),
        radial-gradient(ellipse 60% 40% at 100% 50%, rgba(80, 180, 255, 0.08), transparent),
        radial-gradient(ellipse 50% 30% at 0% 80%, rgba(150, 100, 255, 0.06), transparent) !important;
}

.block-container {
    padding: 1rem 5% 4rem !important;
    max-width: 1200px !important;
}

/* Hero - Bold */
.peak-hero {
    text-align: center;
    padding: 2.5rem 1rem 3rem;
}
.peak-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 6vw, 4rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #fff 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.peak-hero p {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 480px;
    margin: 1rem auto 0;
    line-height: 1.6;
}

/* Stats pill bar */
.peak-stats {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 2rem;
}
.peak-stat {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    color: #64748b;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 0.5rem 1rem;
    border-radius: 100px;
}
.peak-stat strong { color: #e2e8f0; }

/* Main content card */
.peak-main {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
}

/* Upload zone - Premium dropzone */
.peak-upload-wrap {
    padding: 2rem;
    min-height: 340px;
}
.peak-dropzone {
    background: rgba(15, 15, 25, 0.6);
    border: 2px dashed rgba(167, 139, 250, 0.35);
    border-radius: 20px;
    padding: 2.5rem;
    min-height: 280px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
}
.peak-dropzone.filled {
    border-style: solid;
    border-color: rgba(56, 189, 248, 0.4);
    background: rgba(56, 189, 248, 0.05);
}
.peak-dropzone .placeholder {
    text-align: center;
}
.peak-dropzone .placeholder-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.6;
}
.peak-dropzone .placeholder-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #64748b;
}
.peak-dropzone .placeholder-hint {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 0.5rem;
}

/* Results panel */
.peak-results {
    padding: 2rem;
    border-left: 1px solid rgba(255,255,255,0.06);
}
@media (min-width: 992px) {
    .peak-split { display: grid; grid-template-columns: 1fr 1fr; }
}

/* Top prediction - Verdict card */
.peak-verdict {
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.12), rgba(56, 189, 248, 0.08));
    border: 1px solid rgba(167, 139, 250, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.peak-verdict-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}
.peak-verdict-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #fff;
    margin: 0.25rem 0 0.5rem;
}
.peak-verdict-bar-wrap {
    height: 8px;
    background: rgba(0,0,0,0.3);
    border-radius: 4px;
    overflow: hidden;
}
.peak-verdict-bar {
    height: 100%;
    background: linear-gradient(90deg, #a78bfa, #38bdf8);
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Other predictions - Compact list */
.peak-list { margin-top: 1rem; }
.peak-pred {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    margin-bottom: 0.5rem;
    border: 1px solid rgba(255,255,255,0.04);
}
.peak-pred-rank {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: #475569;
    min-width: 28px;
}
.peak-pred-name { font-size: 0.9rem; color: #e2e8f0; flex: 1; }
.peak-pred-pct { font-size: 0.85rem; color: #94a3b8; font-weight: 500; }
.peak-pred-mini-bar {
    flex: 0 0 60px;
    height: 4px;
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
    overflow: hidden;
}
.peak-pred-mini-fill {
    height: 100%;
    background: linear-gradient(90deg, #64748b, #94a3b8);
    border-radius: 2px;
}

/* Advisory - Report style */
.peak-advisory {
    margin-top: 1.5rem;
    padding: 1.5rem;
    background: rgba(34, 197, 94, 0.06);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 16px;
    border-left: 4px solid #22c55e;
}
.peak-advisory-head {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}
.peak-advisory-body {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.7;
}

/* Empty state */
.peak-empty {
    padding: 3rem 2rem;
    text-align: center;
    color: #475569;
    font-size: 0.95rem;
    border: 2px dashed rgba(255,255,255,0.06);
    border-radius: 20px;
}

/* File uploader blend */
.stFileUploader { background: transparent !important; }
.stFileUploader section {
    background: rgba(15, 15, 25, 0.5) !important;
    border: 2px dashed rgba(167, 139, 250, 0.3) !important;
    border-radius: 16px !important;
}

/* Animations */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.peak-main { animation: fadeUp 0.5s ease; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# MODEL
# ------------------------------------------------
@st.cache_resource
def get_model():
    return load_model("model/best_model.pth")

model = get_model()

# ------------------------------------------------
# HERO
# ------------------------------------------------
st.markdown("""
<div class="peak-hero">
    <h1>AgriVision AI</h1>
    <p>Upload a leaf image. Get instant diagnosis, top-5 predictions, and treatment advisories—powered by deep learning.</p>
</div>
""", unsafe_allow_html=True)

# Stats pill
st.markdown("""
<div class="peak-stats">
    <span class="peak-stat">Model: <strong>ResNet18</strong></span>
    <span class="peak-stat">Classes: <strong>17</strong></span>
    <span class="peak-stat">Input: <strong>224×224</strong></span>
    <span class="peak-stat">Output: <strong>Top-5</strong></span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# MAIN CARD - Upload + Results
# ------------------------------------------------
st.markdown('<div class="peak-main">', unsafe_allow_html=True)
col_upload, col_results = st.columns([1, 1])

with col_upload:
    st.markdown('<div class="peak-upload-wrap">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "png", "jpeg", "webp", "heic", "heif"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file:
        try:
            raw_bytes = uploaded_file.getvalue()
            if not raw_bytes:
                st.error("File appears empty.")
                st.stop()
            buf = io.BytesIO(raw_bytes)
            buf.seek(0)
            try:
                pil_img = Image.open(buf).copy()
                pil_img.load()
            except Exception:
                try:
                    import imageio.v3 as iio
                    arr = iio.imread(raw_bytes)
                    pil_img = Image.fromarray(arr).convert("RGB")
                except Exception:
                    raise
            pil_img = ImageOps.exif_transpose(pil_img)
            image = pil_img.convert("RGB")
            st.markdown('<div class="peak-dropzone filled">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Invalid image. ({type(e).__name__})")
            st.stop()
    else:
        st.markdown("""
        <div class="peak-dropzone">
            <div class="placeholder">
                <div class="placeholder-icon">📷</div>
                <div class="placeholder-text">Drop your leaf image here</div>
                <div class="placeholder-hint">JPG, PNG, WebP, HEIC</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_results:
    if uploaded_file and image is not None:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs[0], dim=0)
            top5_prob, top5_indices = torch.topk(probabilities, 5)

        top_name = CLASS_NAMES[top5_indices[0].item()]
        top_conf = top5_prob[0].item()
        top_pct = int(top_conf * 100)

        st.markdown(f"""
        <div class="peak-verdict">
            <div class="peak-verdict-label">Top Prediction</div>
            <div class="peak-verdict-name">{top_name}</div>
            <div class="peak-verdict-bar-wrap">
                <div class="peak-verdict-bar" style="width:{top_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="peak-list">', unsafe_allow_html=True)
        for i in range(1, 5):
            cn = CLASS_NAMES[top5_indices[i].item()]
            cf = top5_prob[i].item()
            pct = int(cf * 100)
            st.markdown(f"""
            <div class="peak-pred">
                <span class="peak-pred-rank">#{i+1}</span>
                <span class="peak-pred-name">{cn}</span>
                <span class="peak-pred-pct">{cf*100:.1f}%</span>
                <div class="peak-pred-mini-bar"><div class="peak-pred-mini-fill" style="width:{pct}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="peak-advisory">
            <div class="peak-advisory-head">Treatment Advisory</div>
            <div class="peak-advisory-body">{DISEASE_INFO[top_name]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="peak-empty">
            Upload an image to see predictions and treatment advice.
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
