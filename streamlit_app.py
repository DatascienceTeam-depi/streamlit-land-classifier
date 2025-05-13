import os
import zipfile
import shutil
import streamlit as st
import gdown
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn

# Set page config
st.set_page_config(page_title="Land Type Classifier", layout="wide")

# Define constants
DATA_DIR = "data/EuroSAT"
MODEL_PATH = "model/resnet18_eurosat.pt"

# Google Drive file IDs
DATA_DRIVE_FILE_ID = "1NiurPYhckTUhVzzIo4hje7DtaAkzCd2B"   # استبدل هذا بـ ID تبع الداتا إذا تغيّر
MODEL_DRIVE_FILE_ID = "1NiurPYhckTUhVzzIo4hje7DtaAkzCd2B"  # استبدل هذا بـ ID تبع النموذج

DATA_DRIVE_URL = f"https://drive.google.com/uc?id={DATA_DRIVE_FILE_ID}"
MODEL_DRIVE_URL = f"https://drive.google.com/uc?id={MODEL_DRIVE_FILE_ID}"

# Download dataset
@st.cache_data(show_spinner=True)
def download_dataset():
    os.makedirs("data", exist_ok=True)
    zip_path = "data/eurosat.zip"

    if not os.path.exists(DATA_DIR):
        st.info("📥 Downloading dataset from Google Drive. Please wait...")
        gdown.download(DATA_DRIVE_URL, zip_path, quiet=False, use_cookies=True)

        st.info("🗂️ Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data")

        os.remove(zip_path)

    return DATA_DIR

# Download model
@st.cache_data(show_spinner=True)
def download_model():
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive. Please wait...")
        gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False, use_cookies=True)
    return MODEL_PATH

# Load model
@st.cache_resource(show_spinner=True)
def load_model():
    download_model()
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1809, 0.1350, 0.1239])
])

# Class labels
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Load dataset and model
data_dir = download_dataset()
model = load_model()

# UI
st.title("🌍 Land Type Classification using Satellite Images")

st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    st.subheader("🔍 Predicted Land Type:")
    st.success(predicted_class)
else:
    st.info("Please upload a satellite image to classify.")

if __name__ == "__main__":
    main()
