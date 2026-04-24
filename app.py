import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
 
st.set_page_config(
    page_title="ADAS Traffic Sign Pipeline",
    page_icon="🚦",
    layout="wide"
)
 
CLASS_NAMES = {
    0:"Speed limit (20km/h)", 1:"Speed limit (30km/h)", 2:"Speed limit (50km/h)",
    3:"Speed limit (60km/h)", 4:"Speed limit (70km/h)", 5:"Speed limit (80km/h)",
    6:"End speed limit (80km/h)", 7:"Speed limit (100km/h)", 8:"Speed limit (120km/h)",
    9:"No passing", 10:"No passing veh>3.5t", 11:"Right-of-way at intersection",
    12:"Priority road", 13:"Yield", 14:"Stop", 15:"No vehicles",
    16:"Veh>3.5t prohibited", 17:"No entry", 18:"General caution",
    19:"Dangerous curve left", 20:"Dangerous curve right", 21:"Double curve",
    22:"Bumpy road", 23:"Slippery road", 24:"Road narrows on right", 25:"Road work",
    26:"Traffic signals", 27:"Pedestrians", 28:"Children crossing",
    29:"Bicycles crossing", 30:"Beware of ice/snow", 31:"Wild animals crossing",
    32:"End speed+passing limits", 33:"Turn right ahead", 34:"Turn left ahead",
    35:"Ahead only", 36:"Go straight or right", 37:"Go straight or left",
    38:"Keep right", 39:"Keep left", 40:"Roundabout mandatory",
    41:"End of no passing", 42:"End no passing veh>3.5t"
}
 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_best.keras")
 
model = load_model()
 
def detect_and_crop_sign(img_bgr, min_area=400):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m_r1 = cv2.inRange(hsv, np.array([0,   70,  50]),  np.array([10,  255, 255]))
    m_r2 = cv2.inRange(hsv, np.array([160, 70,  50]),  np.array([180, 255, 255]))
    m_b  = cv2.inRange(hsv, np.array([100, 80,  50]),  np.array([130, 255, 255]))
    m_y  = cv2.inRange(hsv, np.array([15,  80,  80]),  np.array([35,  255, 255]))
    mask = m_r1 | m_r2 | m_b | m_y
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    valid = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.4 < w / (h + 1e-6) < 2.5:
            valid.append((area, x, y, w, h))
    if not valid:
        return None, None
    valid.sort(reverse=True)
    _, x, y, w, h = valid[0]
    pad = int(0.1 * max(w, h))
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(img_bgr.shape[1], x+w+pad), min(img_bgr.shape[0], y+h+pad)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(img_rgb[y1:y2, x1:x2], (64, 64)), (x1, y1, x2-x1, y2-y1)
 
# ── Page Layout ───────────────────────────────────────────────
st.title(" ADAS Traffic Sign Detection & Classification")
 
st.info(
    "**Step 1 — Sign Detection**  "
    "OpenCV HSV colour-thresholding locates the traffic sign region in the full scene.\\n\\n"
    "**Step 2 — Classification**  "
    "MobileNetV2 (fine-tuned on GTSRB) identifies the sign from 43 possible categories."
)
 
uploaded = st.file_uploader(
    "Upload a full road scene image (JPG / PNG)", type=["jpg", "jpeg", "png"]
)
 
if uploaded:
    data    = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
 
    with st.spinner(" Running two-stage ADAS pipeline..."):
        cropped, bbox = detect_and_crop_sign(img_bgr)
 
    col1, col2, col3 = st.columns(3)
 
    with col1:
        st.subheader(" Stage 1 — Original Scene + BBox")
        if bbox:
            x, y, w, h = bbox
            disp = img_rgb.copy()
            cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 3)
            st.image(disp, use_column_width=True)
            st.success(" Sign region detected")
        else:
            st.image(img_rgb, use_column_width=True)
            st.warning(" No sign detected. Try an image with a clearly visible "
                       "red, blue, or yellow sign.")
 
    with col2:
        st.subheader(" Cropped Sign (64×64)")
        if cropped is not None:
            st.image(cropped, width=200)
            st.caption("Resized to 64×64 — classifier input")
        else:
            st.error("Detection failed — no valid crop available.")
 
    with col3:
        st.subheader(" Stage 2 — Classification Result")
        if cropped is not None:
            inp   = preprocess_input(cropped.astype(np.float32))
            probs = model.predict(np.expand_dims(inp, 0), verbose=0)[0]
            top3  = probs.argsort()[-3:][::-1]
 
            pred_name  = CLASS_NAMES[top3[0]]
            confidence = probs[top3[0]] * 100
 
            st.metric("Predicted Sign", pred_name)
            st.metric("Confidence",     f"{confidence:.1f}%")
 
            st.markdown("**Top-3 Predictions:**")
            for idx in top3:
                st.progress(
                    float(probs[idx]),
                    text=f"{CLASS_NAMES[idx]} — {probs[idx]*100:.1f}%"
                )
        else:
            st.info("Awaiting valid detection result.")
 
st.markdown("---")
st.caption(
    "Model: MobileNetV2 fine-tuned on GTSRB (43 classes) | "
    "Detection: OpenCV HSV colour thresholding + contour filtering | "
    "Project: HCL GUVI Module 23"
)
