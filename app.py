import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import requests
import os
import io
import base64

# --- Configuration and File Paths ---
IMAGE_SIZE = 256
MODEL_PATH = 'deeplabv3_best_lr_5e-05.pth'
HF_MODEL_URL = "https://huggingface.co/Sighakolli-2Mounica/Vision_Extract/resolve/main/deeplabv3_best_lr_5e-05.pth"
DEVICE = torch.device("cpu")

# --- Custom Style ---
def set_custom_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        .stApp { background: #f0f8ff !important; }
        h1 { text-align: center; font-weight: 700; color: #2c3e50; margin-bottom: 0.3em; }
        .app-description { text-align: center; font-size: 20px; line-height: 1.6; color: #34495e; margin-bottom: 1em; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        iframe { margin-bottom: 0px !important; }
        .image-frame { border-radius: 16px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .stButton button {
            background: #0d47a1; color: white !important; border-radius: 12px;
            padding: 12px 25px; font-size: 16px; font-weight: bold;
            border: none; transition: all 0.3s ease-in-out;
        }
        .stButton button:hover { transform: scale(1.05); background: #1565c0; }
        section[data-testid="stFileUploader"] {
            border: 2px dashed #3498db; border-radius: 12px; background-color: rgba(236,247,255,0.7);
        }
        section[data-testid="stFileUploader"]:hover {
            border-color: #2575fc; background-color: rgba(200,230,255,0.9);
        }
        /* hide juxtapose watermark */
        .juxtapose-credit { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Helper Functions ---
def get_image_download_link_button(img_array, filename):
    img_array_uint8 = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array_uint8)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return st.download_button(
        label="Download Masked Image",
        data=buffered.getvalue(),
        file_name=f"{filename}.png",
        mime="image/png",
        key='download_mask_btn'
    )

resize_transform = T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.NEAREST)
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

@st.cache_resource
def download_model_file():
    try:
        response = requests.get(HF_MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False

@st.cache_resource
def load_segmentation_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    except Exception:
        return None
    model.to(DEVICE)
    model.eval()
    return model

def run_segmentation_pipeline(image_pil, model):
    image_tensor = T.ToTensor()(resize_transform(image_pil))
    image_tensor = normalize(image_tensor).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)['out']
    probabilities = torch.sigmoid(output)
    predicted_mask = (probabilities > 0.5).float().squeeze(0).cpu().numpy().squeeze()
    un_normalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225])
    original_image_np = un_normalize(image_tensor.squeeze(0)).cpu().permute(1, 2, 0).numpy()
    original_image_np = np.clip(original_image_np, 0, 1)
    masked_result = np.zeros_like(original_image_np)
    masked_result[predicted_mask.astype(bool)] = original_image_np[predicted_mask.astype(bool)]
    return original_image_np, masked_result

# --- Slider for demo section ---
def custom_image_comparison(img1: Image.Image, img2: Image.Image):
    def pil_to_base64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    img1_b64 = pil_to_base64(img1)
    img2_b64 = pil_to_base64(img2)

    html_code = f"""
    <link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">
    <script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>
    <div id="juxtapose-wrapper" style="max-width:350px; margin:auto;"></div>
    <script>
      new juxtapose.JXSlider('#juxtapose-wrapper',
        [
          {{ src: "data:image/png;base64,{img1_b64}", label: "Original" }},
          {{ src: "data:image/png;base64,{img2_b64}", label: "Masked" }}
        ],
        {{ animate: true, showLabels: true, showCredits: false, startingPosition: "50%" }}
      );
    </script>
    """
    st.components.v1.html(html_code, height=300)

# --- Side-by-side results for user uploads ---
def show_uploaded_results(original, masked):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='text-align:center;'>Original Image</h4>", unsafe_allow_html=True)
        st.image(original, use_container_width=True)
    with col2:
        st.markdown("<h4 style='text-align:center;'>Masked Image</h4>", unsafe_allow_html=True)
        st.image(masked, use_container_width=True)
        get_image_download_link_button(masked, "segmented_result")

# --- UI ---
set_custom_style()

if not os.path.exists(MODEL_PATH):
    if not download_model_file():
        st.stop()

segmentation_model = load_segmentation_model()

# Title & description
st.title("Vision Extract")
st.markdown(
    """
    <p class="app-description">
    <span style="color:#2575fc; font-weight:600;">Vision Extract</span> is an 
    <span style="color:#27ae60; font-weight:600;">AI-powered tool</span> that instantly 
    separates objects from their backgrounds.  
    Simply upload an image, and the system generates a clean cut-out mask you can use for 
    <b>editing</b>, <b>presentations</b>, or <b>creative projects</b>.  
    It’s <span style="color:#e67e22; font-weight:600;">fast</span>, 
    <span style="color:#e67e22; font-weight:600;">accurate</span>, and built to make 
    <b>image extraction effortless</b>.
    </p>
    """,
    unsafe_allow_html=True
)


# --- Demo Slider (Ad section) ---
try:
    orig_demo = Image.open("sample_original_image.png")
    mask_demo = Image.open("sample_masked_resized.png")
    st.markdown("### See it in Action 👇")
    custom_image_comparison(orig_demo, mask_demo)
except Exception:
    st.info("⚠️ Demo images not found. Please add `sample_original_image.png` and `sample_masked_resized.png`.")

# --- Upload Section ---
if segmentation_model is not None:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an Image to Analyze", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    if uploaded_file is not None:
        with st.spinner("✨ Analyzing image and running DeepLabV3 inference..."):
            original_image_pil = Image.open(uploaded_file).convert("RGB")
            original_image_np, masked_image_np = run_segmentation_pipeline(original_image_pil, segmentation_model)

        # Show results side by side
        st.subheader("Segmentation Result")
        show_uploaded_results(original_image_np, masked_image_np)
