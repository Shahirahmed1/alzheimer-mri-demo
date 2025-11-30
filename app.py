# app.py (polished demo with Grad-CAM + sample images + auto-loader)
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import os
from gradcam_utils import load_baseline, load_resnet, gradcam_for_image, DEVICE

st.set_page_config(page_title="Alzheimer MRI Demo", layout="wide")
st.title("Alzheimer's MRI — Live Demo")
st.markdown("Upload a 2D MRI slice (PNG/JPG). The app returns prediction and Grad-CAM overlay.")

# Sidebar controls
st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("Model", ["Baseline CNN", "ResNet18", "Ensemble"])
show_prob_table = st.sidebar.checkbox("Show probability table", value=True)
use_samples = st.sidebar.selectbox("Try sample image", ["None", "Sample 1", "Sample 2", "Sample 3"])
st.sidebar.markdown("---")
st.sidebar.write("Models loaded from repo `models/` folder.")

# Lazy load models
@st.cache_resource
def load_models():
    # paths: adjust if you host models externally
    baseline_path = "models/cnn_baseline_fixed.pth"
    resnet_path = "models/resnet18_adapted.pth"
    baseline = load_baseline(baseline_path)
    resnet = load_resnet(resnet_path)
    return baseline, resnet

baseline, resnet = load_models()

CLASS_NAMES = ["Mild_Demented","Moderate_Demented","Non_Demented","Very_Mild_Demented"]

# Samples (replace with real sample images in repo under /samples/)
SAMPLES = {
    "Sample 1": "samples/sample1.png",
    "Sample 2": "samples/sample2.png",
    "Sample 3": "samples/sample3.png"
}

col_left, col_right = st.columns([1,2])

with col_left:
    uploaded = st.file_uploader("Upload MRI image (PNG/JPG)", type=["png","jpg","jpeg"])
    if use_samples != "None":
        path = SAMPLES.get(use_samples)
        if path and os.path.exists(path):
            uploaded = open(path, "rb")

    if uploaded:
        if hasattr(uploaded, "read"):
            img = Image.open(uploaded).convert("L")
        else:
            img = Image.open(uploaded).convert("L")
        st.image(img, caption="Input image", use_column_width=True)
    else:
        st.info("Upload an MRI slice or pick a sample.")

with col_right:
    if uploaded:
        # prediction
        tensor = torch.tensor(np.array(img)).unsqueeze(0).unsqueeze(0).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(DEVICE)
        with st.spinner("Predicting..."):
            out_b = baseline(tensor)
            out_r = resnet(tensor)
            probs_b = F.softmax(out_b, dim=1).cpu().numpy()[0]
            probs_r = F.softmax(out_r, dim=1).cpu().numpy()[0]
            if model_choice == "Baseline CNN":
                probs = probs_b
            elif model_choice == "ResNet18":
                probs = probs_r
            else:
                probs = (probs_b + probs_r) / 2.0

            pred_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[pred_idx]
            st.subheader(f"Prediction: {pred_name} ({probs[pred_idx]:.3f})")

            if show_prob_table:
                st.table({CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))})

        # Grad-CAM
        with st.spinner("Generating Grad-CAM..."):
            if model_choice == "Baseline CNN":
                cam, cls, _ = gradcam_for_image(baseline, img, target_class=None)
                overlay = create_overlay_image(img, cam)
                st.image(overlay, caption="Grad-CAM (Baseline)", use_column_width=True)
            elif model_choice == "ResNet18":
                cam, cls, _ = gradcam_for_image(resnet, img, target_class=None)
                overlay = create_overlay_image(img, cam)
                st.image(overlay, caption="Grad-CAM (ResNet)", use_column_width=True)
            else:
                cam_b, cls_b, _ = gradcam_for_image(baseline, img, target_class=None)
                cam_r, cls_r, _ = gradcam_for_image(resnet, img, target_class=None)
                st.image([create_overlay_image(img, cam_b), create_overlay_image(img, cam_r)],
                         caption=["Baseline Grad-CAM","ResNet Grad-CAM"], width=350)

# small helper (placed after imports)
def create_overlay_image(pil_orig, cam, alpha=0.5):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    cam_img = Image.fromarray(np.uint8(cam*255)).resize(pil_orig.size)
    heat = plt.get_cmap("jet")(np.array(cam_img)/255.0)[..., :3]
    base = pil_orig.convert("RGB")
    overlay = (0.5*np.array(base) + 0.5*(heat*255)).astype("uint8")
    return Image.fromarray(overlay)
