"""
Handwritten Digit Recognition — Streamlit App
Run: streamlit run app.py
Requires: mnist_cnn.h5 (produced by the Jupyter notebook)
"""


import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import tensorflow as tf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Digit Recognition", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .title-block {
    background: linear-gradient(135deg,#0d1b2a,#1a3a5c);
    padding:2rem 2.5rem; border-radius:14px;
    margin-bottom:1.5rem; text-align:center;
  }
  .title-block h1 { color:#e8eaf6; font-size:2rem; font-weight:700; margin:0; }
  .title-block p  { color:#90a4ae; margin:0.4rem 0 0; font-size:0.95rem; }

  .result-card {
    background:#fff; border-left:6px solid #1a3a5c;
    border-radius:10px; padding:1.2rem 1.5rem;
    margin:0.8rem 0; box-shadow:0 2px 10px rgba(0,0,0,.08);
  }
  .result-digit { font-size:4.5rem; font-weight:900; color:#1a3a5c; line-height:1; }
  .lbl { font-size:.75rem; color:#78909c; text-transform:uppercase; letter-spacing:1px; }
  .conf-high   { color:#2e7d32; font-weight:700; font-size:1.15rem; }
  .conf-medium { color:#e65100; font-weight:700; font-size:1.15rem; }
  .conf-low    { color:#c62828; font-weight:700; font-size:1.15rem; }

  .sec { font-size:.75rem; font-weight:700; color:#455a64;
         text-transform:uppercase; letter-spacing:1px;
         margin:1rem 0 .4rem; border-bottom:1px solid #cfd8dc; padding-bottom:.3rem; }

  .prob-row  { display:flex; align-items:center; margin:4px 0; gap:8px; }
  .prob-lbl  { width:16px; font-weight:700; font-size:.9rem; text-align:right; color:#263238; }
  .prob-bg   { flex:1; background:#eceff1; border-radius:4px; height:20px; overflow:hidden; }
  .prob-fill { height:100%; border-radius:4px; }
  .prob-pct  { width:46px; font-size:.78rem; color:#607d8b; text-align:right; }
  .warn      { background:#fff8e1; border-left:4px solid #f9a825;
               border-radius:6px; padding:.7rem 1rem; margin:.5rem 0;
               font-size:.85rem; color:#5d4037; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "mnist_cnn.h5"   # produced by the notebook
IMG_SIZE   = 28               # model expects 28×28×1


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


# ── Preprocessing — MNIST-style normalisation ─────────────────────────────────
def preprocess(pil_img: Image.Image):
    """
    Converts any real-world image into clean MNIST format:
      white digit on black background, tight crop, centred, 28×28 grayscale.

    Returns
    -------
    batch   : np.ndarray  shape (1, 28, 28, 1)  float32 in [0,1]
    preview : PIL.Image   what the model actually receives
    """

    # 1. Grayscale
    gray = pil_img.convert("L")

    # 2. Boost contrast so faint pencil strokes become visible
    gray = ImageEnhance.Contrast(gray).enhance(3.0)

    # 3. Convert to array
    arr = np.array(gray, dtype=np.float32)   # 0–255

    # 4. Invert if background is light (camera/scanner: dark ink on white paper)
    #    MNIST convention: white digit on BLACK background
    if arr.mean() > 127:
        arr = 255.0 - arr

    # 5. Denoise
    tmp = Image.fromarray(arr.astype(np.uint8))
    tmp = tmp.filter(ImageFilter.MedianFilter(size=3))
    arr = np.array(tmp, dtype=np.float32)

    # 6. Adaptive binary threshold — keep only the digit strokes
    flat   = arr.flatten()
    bright = flat[flat > 30]
    thresh = float(np.percentile(bright, 25)) if len(bright) > 100 else 40.0
    binary = np.where(arr >= thresh, arr, 0.0)   # soft threshold (not hard 0/255)

    # 7. Tight bounding-box crop around non-zero pixels
    mask = binary > 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        digit  = binary[r0:r1+1, c0:c1+1]
    else:
        digit = binary

    # 8. Make square canvas with 20 % padding — replicates MNIST centering
    h, w  = digit.shape
    side  = max(h, w)
    sq    = np.zeros((side, side), dtype=np.float32)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    sq[y_off:y_off+h, x_off:x_off+w] = digit

    pad     = int(side * 0.20)
    padded  = np.zeros((side + pad*2, side + pad*2), dtype=np.float32)
    padded[pad:pad+side, pad:pad+side] = sq

    # 9. Light Gaussian smooth before resize (avoids jagged edges)
    tmp2    = Image.fromarray(padded.astype(np.uint8))
    tmp2    = tmp2.filter(ImageFilter.GaussianBlur(radius=0.5))
    padded  = np.array(tmp2, dtype=np.float32)

    # 10. Resize to 28×28 (model input size)
    pil_28  = Image.fromarray(padded.astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS
    )
    preview = pil_28.copy()

    # 11. Normalise to [0,1] and add batch + channel dims
    arr_28  = np.array(pil_28, dtype=np.float32) / 255.0
    batch   = arr_28[np.newaxis, :, :, np.newaxis]   # (1,28,28,1)

    return batch, preview


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model, batch):
    probs = model.predict(batch, verbose=0)[0]   # shape (10,)
    digit = int(np.argmax(probs))
    conf  = float(probs[digit]) * 100
    return digit, conf, probs


# ── Helpers ───────────────────────────────────────────────────────────────────
def conf_cls(c):
    return "conf-high" if c >= 80 else "conf-medium" if c >= 55 else "conf-low"

def prob_bars_html(probs, predicted):
    html = ""
    for i, p in enumerate(probs):
        pct   = p * 100
        w     = max(pct, 0.5)
        color = "#1a3a5c" if i == predicted else "#b0bec5"
        html += (
            f'<div class="prob-row">'
            f'<div class="prob-lbl">{i}</div>'
            f'<div class="prob-bg"><div class="prob-fill" '
            f'style="width:{w:.1f}%;background:{color};"></div></div>'
            f'<div class="prob-pct">{pct:.1f}%</div></div>'
        )
    return html


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.write(
        "Custom Deep CNN trained on 60,000 MNIST samples. "
        "Expected accuracy: **99.4–99.6%** on the test set."
    )
    st.divider()
    st.markdown("### Tips for best accuracy")
    st.markdown(
        "- One digit per image\n"
        "- Dark ink on white/light background\n"
        "- Thick, clear strokes\n"
        "- Digit fills most of the frame\n"
        "- No shadows, no background patterns\n"
        "- Photograph straight-on, not at an angle"
    )
    st.divider()
    st.markdown("### Model details")
    st.markdown(
        "Architecture: 3-block Deep CNN  \n"
        "Input: 28 × 28 grayscale  \n"
        "Trained: 50 epochs max with early stopping  \n"
        "File: `mnist_cnn.h5`"
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>Handwritten Digit Recognition</h1>
  <p>Upload a photo or scan of a single handwritten digit — the model will identify it</p>
</div>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)

if model is None:
    st.error(
        f"**Model file `{MODEL_PATH}` not found.**\n\n"
        "Steps to fix:\n"
        "1. Open `mnist_digit_recognition.ipynb` in Jupyter\n"
        "2. Run **all cells** (Kernel → Restart & Run All)\n"
        "3. Wait for training to finish — it saves `mnist_cnn.h5` automatically\n"
        "4. Make sure `mnist_cnn.h5` is in the **same folder** as `app.py`\n"
        "5. Restart this app"
    )
    st.stop()

st.success("Model loaded — ready to predict")
st.markdown("---")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Upload Image", "Random Test"])

# ── Tab 1: Upload ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec">Choose an image file</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "PNG, JPG or BMP — one digit per image",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        pil_img = Image.open(uploaded)

        with st.spinner("Preprocessing and predicting..."):
            try:
                batch, preview     = preprocess(pil_img)
                digit, conf, probs = predict(model, batch)
                ok = True
            except Exception as e:
                ok = False
                err = str(e)

        if not ok:
            st.error(f"Prediction failed: {err}")
            st.stop()

        # ── Three columns: original | processed | result ──────────────────
        c1, c2, c3 = st.columns([1, 1, 1.3], gap="medium")

        with c1:
            st.markdown('<div class="sec">Original</div>', unsafe_allow_html=True)
            st.image(pil_img, use_column_width=True)
            st.caption(f"{pil_img.size[0]}×{pil_img.size[1]}  {pil_img.mode}")

        with c2:
            st.markdown('<div class="sec">Model input (28×28)</div>', unsafe_allow_html=True)
            # Display preview at larger size for readability
            preview_large = preview.resize((140, 140), Image.NEAREST)
            st.image(preview_large, use_column_width=False, width=140)
            st.caption("MNIST-normalised grayscale")

            # Warn if the processed image looks blank
            prev_arr = np.array(preview)
            if prev_arr.mean() < 8:
                st.markdown(
                    '<div class="warn">Processed image looks blank — '
                    'digit may not have been detected. '
                    'Try a clearer photo with better contrast.</div>',
                    unsafe_allow_html=True
                )

        with c3:
            st.markdown('<div class="sec">Prediction</div>', unsafe_allow_html=True)
            cc = conf_cls(conf)
            st.markdown(f"""
            <div class="result-card">
              <div class="lbl">Predicted digit</div>
              <div class="result-digit">{digit}</div>
              <div style="margin-top:.5rem;">
                <span class="lbl">Confidence &nbsp;</span>
                <span class="{cc}">{conf:.1f}%</span>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec">All probabilities</div>', unsafe_allow_html=True)
            st.markdown(prob_bars_html(probs, digit), unsafe_allow_html=True)

        # Low-confidence warning below all columns
        if conf < 60:
            st.markdown(
                '<div class="warn">Confidence is low. The model is uncertain. '
                'Try a cleaner image: single digit, thick dark strokes, '
                'plain white background, no shadows, photo taken straight-on.</div>',
                unsafe_allow_html=True
            )


# ── Tab 2: CSV sample ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec">Random sample from test set</div>', unsafe_allow_html=True)

    if st.button("Load random sample"):
        try:
            import pandas as pd
            df  = pd.read_csv("mnist_test.csv")
            row = df.sample(1, random_state=None).iloc[0]
            true_label = int(row["label"])
            pixels = row.drop("label").values.astype(np.float32) / 255.0
            img28  = pixels.reshape(28, 28)

            # Build PIL from raw 28×28
            pil_s  = Image.fromarray((img28 * 255).astype(np.uint8), mode="L")

            # Predict directly without heavy preprocessing (already MNIST format)
            arr_s  = img28[np.newaxis, :, :, np.newaxis].astype(np.float32)
            digit, conf, probs = predict(model, arr_s)

            c1, c2, c3 = st.columns([1, 1, 1.3], gap="medium")

            with c1:
                st.markdown('<div class="sec">Original 28×28</div>', unsafe_allow_html=True)
                st.image(pil_s.resize((140,140), Image.NEAREST), width=140)
                st.caption(f"True label: **{true_label}**")

            with c2:
                st.markdown('<div class="sec">As fed to model</div>', unsafe_allow_html=True)
                st.image(pil_s.resize((140,140), Image.NEAREST), width=140)
                st.caption("No preprocessing needed (already MNIST format)")

            with c3:
                st.markdown('<div class="sec">Prediction</div>', unsafe_allow_html=True)
                correct    = digit == true_label
                card_color = "#1b5e20" if correct else "#b71c1c"
                verdict    = "Correct" if correct else "Wrong"
                cc = conf_cls(conf)
                st.markdown(f"""
                <div class="result-card" style="border-left-color:{card_color};">
                  <div class="lbl">Predicted digit</div>
                  <div class="result-digit" style="color:{card_color};">{digit}</div>
                  <div style="margin-top:.4rem;">
                    <span class="lbl">Confidence &nbsp;</span>
                    <span class="{cc}">{conf:.1f}%</span>
                  </div>
                  <div style="margin-top:.3rem;">
                    <span class="lbl">Verdict &nbsp;</span>
                    <span style="color:{card_color};font-weight:700;">{verdict}</span>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.markdown('<div class="sec">All probabilities</div>', unsafe_allow_html=True)
                st.markdown(prob_bars_html(probs, digit), unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("mnist_test.csv not found — place it in the same folder as app.py.")
        except Exception as e:
            st.error(f"Error: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#90a4ae;font-size:.78rem;'>"
    "Handwritten Digit Recognition</p>",
    unsafe_allow_html=True
)
