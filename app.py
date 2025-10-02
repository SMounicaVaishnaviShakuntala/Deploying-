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
import time
import uuid

# --- Initialize Gallery in Session State ---
if "gallery" not in st.session_state:
    st.session_state["gallery"] = []

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
         .block-container {
            max-width: 100% !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Align text blocks properly */
        /* Keep title + description centered */
    h1, .app-description {
        text-align: center !important;
    }

    /* Make all other section text left-aligned */
    h2, h3, p {
        text-align: left !important;
    }


        /* Ensure images and columns expand fully */
        .stImage {
            width: 60% !important;
        }
        /* hide juxtapose watermark */
        .juxtapose-credit { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Helper Functions ---
def get_image_download_link_button(img, filename, key: str = None):
    """
    Create a Streamlit download_button for a PIL image or numpy array.
    - img: PIL.Image.Image or numpy array (range 0..1 or uint8 0..255).
    - filename: base filename without extension.
    - key: optional unique Streamlit key. If None, a uuid4 key is generated.
    """
    # Generate unique key if not provided
    if key is None:
        key = f"download_{uuid.uuid4().hex}"

    # Convert to PIL if needed
    if isinstance(img, Image.Image):
        pil_img = img
    else:
        arr = np.array(img)
        # If float image in 0..1, convert to uint8
        if arr.dtype != np.uint8:
            try:
                arr = (arr * 255).astype(np.uint8)
            except Exception:
                arr = arr.astype(np.uint8)
        pil_img = Image.fromarray(arr)

    # Save to buffer and create download_button
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    buffered.seek(0)
    return st.download_button(
        label="📥 Download Masked Image",
        data=buffered.getvalue(),
        file_name=f"{filename}.png",
        mime="image/png",
        key=key
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
    st.components.v1.html(html_code, height=330)

def show_uploaded_results(original, masked):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4 style='text-align:center;'>Original Image</h4>", unsafe_allow_html=True)
        st.image(original, use_container_width=True)

    with col2:
        st.markdown("<h4 style='text-align:center;'>Masked Image</h4>", unsafe_allow_html=True)
        st.image(masked, use_container_width=True)

        # ✅ Proper centering using empty columns
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 2])  
        with btn_col2:
            get_image_download_link_button(masked, "segmented_result")

def feedback_section():
    st.markdown("### Share your feedback 😊", unsafe_allow_html=True)

    feedback_html = """
    <style>
    .emoji-slider {
        display: flex;
        justify-content: center;
        align-items: flex-end;
        gap: 25px;
        margin: 20px auto;
    }
    .emoji-slider .emoji-wrapper {
        position: relative;
        text-align: center;
    }
    .emoji-slider span {
        font-size: 35px;
        cursor: pointer;
        transition: transform 0.2s, background 0.2s;
        display: inline-block;
        padding: 10px;
        border-radius: 50%;
    }
    .emoji-slider span:hover {
        transform: scale(1.4);
        background: rgba(0,0,0,0.05);
    }
    /* Tooltip */
    .emoji-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: 50px;
        left: 50%;
        transform: translateX(-50%);
        background: #2c3e50;
        color: white;
        padding: 5px 10px;
        border-radius: 8px;
        font-size: 13px;
        white-space: nowrap;
        transition: opacity 0.3s;
    }
    .emoji-wrapper:hover .emoji-tooltip {
        visibility: visible;
        opacity: 1;
    }
    .thank-you {
        color: #27ae60;
        font-weight: bold;
        text-align: center;
        font-size: 17px;
        margin-top: 15px;
    }
    </style>

    <div class="emoji-slider" id="emoji-slider">
        <div class="emoji-wrapper">
            <span id="1">😡</span>
            <div class="emoji-tooltip">Very Dissatisfied</div>
        </div>
        <div class="emoji-wrapper">
            <span id="2">🙁</span>
            <div class="emoji-tooltip">Dissatisfied</div>
        </div>
        <div class="emoji-wrapper">
            <span id="3">😐</span>
            <div class="emoji-tooltip">Neutral</div>
        </div>
        <div class="emoji-wrapper">
            <span id="4">😊</span>
            <div class="emoji-tooltip">Satisfied</div>
        </div>
        <div class="emoji-wrapper">
            <span id="5">😍</span>
            <div class="emoji-tooltip">Very Satisfied</div>
        </div>
    </div>

    <p id="feedback-message" class="thank-you"></p>

    <script>
    const emojis = document.querySelectorAll('#emoji-slider span');
    const msg = document.getElementById('feedback-message');

    emojis.forEach(emoji => {
        emoji.addEventListener('click', () => {
            msg.textContent = "🎉 Thanks for your feedback!";
        });
    });
    </script>
    """

    st.components.v1.html(feedback_html, height=220)


# --- UI ---

set_custom_style()

if not os.path.exists(MODEL_PATH):
    if not download_model_file():
        st.stop()

segmentation_model = load_segmentation_model()

# Title & description
st.title("VisionExtract — AI-Powered Subject Isolation")
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
# --- Upload Section ---
if segmentation_model is not None:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("✨ Upload an image to see instant subject isolation", type=["jpg", "jpeg", "png"], key="file_uploader")
    st.markdown("---")

    if uploaded_file is not None:
        progress = st.progress(0)
        status_text = st.empty()

        # Step 1: Uploading
        status_text.text("📤 Uploading image...")
        time.sleep(0.5)
        progress.progress(25)

        original_image_pil = Image.open(uploaded_file).convert("RGB")

        # Step 2: Running AI model
        status_text.text("🤖 Running AI model...")
        time.sleep(0.5)
        progress.progress(50)

        # Run segmentation
        original_image_np, masked_image_np = run_segmentation_pipeline(original_image_pil, segmentation_model)

        # Step 3: Generating mask
        status_text.text("🎨 Generating mask...")
        time.sleep(0.5)
        progress.progress(75)

        # Step 4: Finalizing
        status_text.text("✅ Finalizing results...")
        time.sleep(0.5)
        progress.progress(100)

        # Clear progress + message after done
        progress.empty()
        status_text.empty()

        # Show results side by side
        st.subheader("Segmentation Result")
        show_uploaded_results(original_image_np, masked_image_np)

        # Feedback section
        feedback_section()

        # --- Save to session gallery ---
        uploaded_name = uploaded_file.name
        # Avoid duplicate entries for the same uploaded file
        already_in_gallery = any(entry["filename"] == uploaded_name for entry in st.session_state["gallery"])

        if not already_in_gallery:
            # Save both original + masked into gallery with metadata
            st.session_state["gallery"].insert(0, {
                "filename": uploaded_name,
                "original": original_image_np,
                "masked": masked_image_np
            })
            # Keep only last 5 uploads
            st.session_state["gallery"] = st.session_state["gallery"][:5]

# --- Mini Gallery Section ---
if "show_gallery" not in st.session_state:
    st.session_state["show_gallery"] = False

def toggle_gallery():
    st.session_state["show_gallery"] = not st.session_state["show_gallery"]

def clear_gallery():
    st.session_state["gallery"] = []
    st.session_state["show_gallery"] = False

if st.session_state["gallery"]:  # Only if there are images
    if not st.session_state["show_gallery"]:
        st.button("📂 Show Gallery", on_click=toggle_gallery, key="show_gallery_btn")
    else:
        st.markdown("### 🖼️ Recent Uploads")

        # Show thumbnails in columns
        cols = st.columns(len(st.session_state["gallery"]))
        for i, entry in enumerate(st.session_state["gallery"]):
            with cols[i]:
                # Resize to thumbnail before rendering (faster)
                thumb_orig = Image.fromarray((entry["original"] * 255).astype(np.uint8)).resize((200, 200))
                thumb_mask = Image.fromarray((entry["masked"] * 255).astype(np.uint8)).resize((200, 200))

                st.image(thumb_orig, caption=f"Original {i+1}")
                st.image(thumb_mask, caption=f"Masked {i+1}")
                get_image_download_link_button(entry["masked"], f"gallery_masked_{i+1}", key=f"gallery_dl_{i}")

        # Action buttons below the gallery
        colA, colB = st.columns([1, 1])
        with colA:
            st.button("🗑️ Clear Gallery", on_click=clear_gallery, key="clear_gallery_btn")
        with colB:
            st.button("❌ Close Gallery", on_click=toggle_gallery, key="close_gallery_btn")
