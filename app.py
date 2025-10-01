import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import requests
import os
from torchvision.transforms import InterpolationMode
import io

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

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        /* 🔹 Page background */
        .stApp {
            background: #f0f8ff !important;  /* Light sky blue */
        }

        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 700;
            text-align: center;
        }

        /* Result cards */
        .result-card {
            border-radius: 20px;
            padding: 15px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            margin-bottom: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .result-card:hover {
            transform: scale(1.03);
            box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        }

        .original-card {
            background: rgba(173, 216, 230, 0.25);
            border: 2px solid #3498db;
        }
        .masked-card {
            background: rgba(144, 238, 144, 0.25);
            border: 2px solid #27ae60;
        }

        .stButton button {
            background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
            color: white !important;
            border-radius: 12px;
            padding: 12px 25px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease-in-out;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }

        .stButton button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #2575fc 0%, #6a11cb 100%);
        }

        /* Confidence metric */
        div[data-testid="stMetricValue"] {
            font-size: 2em;
            color: #27ae60;
            font-weight: bold;
            text-shadow: none; /* removed glowing */
        }

        /* File uploader */
        section[data-testid="stFileUploader"] {
            border: 2px dashed #3498db;
            border-radius: 12px;
            background-color: rgba(236,247,255,0.7);
            transition: all 0.3s ease;
        }
        section[data-testid="stFileUploader"]:hover {
            border-color: #2575fc;
            background-color: rgba(200,230,255,0.9);
        }

        /* 🔍 Make zoom/magnifier icon bigger + visible */
        button[title="View fullscreen"] {
            transform: scale(1.6); /* bigger icon */
            filter: brightness(0) invert(1); /* white on dark */
            background: rgba(0,0,0,0.4);
            border-radius: 50%;
        }
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

resize_transform = T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.NEAREST)
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

    model = deeplabv3_resnet50(weights=None) 
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        )
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

    probabilities_np = probabilities.cpu().numpy().squeeze()
    foreground_probabilities = probabilities_np[predicted_mask.astype(bool)]
    avg_confidence = np.mean(foreground_probabilities) if foreground_probabilities.size > 0 else 0.0

    un_normalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225])
    original_image_np = un_normalize(image_tensor.squeeze(0)).cpu().permute(1, 2, 0).numpy()
    original_image_np = np.clip(original_image_np, 0, 1)

    masked_result = np.zeros_like(original_image_np)
    masked_result[predicted_mask.astype(bool)] = original_image_np[predicted_mask.astype(bool)]

    return original_image_np, masked_result, avg_confidence

# --- UI ---
set_custom_style()

if not os.path.exists(MODEL_PATH):
    if not download_model_file():
        st.stop()

segmentation_model = load_segmentation_model()

st.title("✂️ Vision Extract Tool")

# 🔹 Interactive description (not in a box, fade-in effect)
st.markdown(
    """
    <div style="text-align: center; margin-top: -10px; margin-bottom: 25px;">
        <p style="
            font-size: 20px; 
            color: #34495e; 
            font-weight: 500; 
            line-height: 1.6; 
            animation: fadeIn 2s ease-in-out;">
            Isolate objects from any image in <b>seconds</b> with our 
            <span style="color:#2575fc; font-weight:600;">DeepLabV3 Segmentation Tool</span>.  
            Powered by <b>advanced AI</b>, it delivers precise, high-confidence masks for 
            editing, research, and real-world applications — from 
            <span style="color:#27ae60; font-weight:600;">photo enhancements</span> 
            to <span style="color:#e67e22; font-weight:600;">self-driving technologies</span>.
        </p>
    </div>

    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True
)

if segmentation_model is not None:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an Image to Analyze", type=["jpg", "jpeg", "png"])
    st.markdown("---")

    if uploaded_file is not None:
        with st.spinner("✨ Analyzing image and running DeepLabV3 inference..."):
            original_image_pil = Image.open(uploaded_file).convert("RGB")
            original_image_np, masked_image_np, avg_confidence = run_segmentation_pipeline(original_image_pil, segmentation_model)

        confidence_text = f"{avg_confidence * 100:.2f}%"

        st.subheader("Segmentation Result")
        output_col1, output_col2 = st.columns(2)

        with output_col1:
            st.markdown('<div class="result-card original-card">', unsafe_allow_html=True)
            st.header("Original Image")
            st.image(original_image_np, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with output_col2:
            st.markdown('<div class="result-card masked-card">', unsafe_allow_html=True)
            st.header("Masked Image")
            st.image(masked_image_np, use_container_width=True)
            st.metric(
                label="Average Foreground Confidence (0-100%)",
                value=confidence_text
            )
            get_image_download_link_button(masked_image_np, "segmented_result")
            st.markdown('</div>', unsafe_allow_html=True)
