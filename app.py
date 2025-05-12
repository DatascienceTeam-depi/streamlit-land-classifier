import os
import zipfile
import requests
import streamlit as st
from PIL import Image
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

st.set_page_config(page_title="Land Type Classification", layout="wide")
st.title("🌍 Land Type Classification using Deep Learning")

@st.cache_resource
def download_dataset():
    url = "https://drive.google.com/uc?export=download&id=1NiurPYhckTUhVzzIo4hje7DtaAkzCd2B"
    zip_path = "data/eurosat.zip"
    extract_path = "data/EuroSAT"

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(zip_path):
        st.info("Downloading dataset...")
        with requests.get(url, stream=True) as r:
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    if not os.path.exists(extract_path):
        st.info("Extracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data/")
        except zipfile.BadZipFile:
            raise Exception("Downloaded file is not a valid ZIP. Please recheck the Google Drive link.")

    return extract_path

# Load dataset and model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)  # EuroSAT has 10 classes
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def get_class_labels():
    return ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
            'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return preds.item()

# Main logic
data_dir = download_dataset()
model = load_model()
class_labels = get_class_labels()

uploaded_file = st.file_uploader("Upload a satellite image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict_image(image, model)
    st.success(f"Predicted Land Type: {class_labels[prediction]}")

# Show some dataset samples
if st.checkbox("Show sample images from dataset"):
    st.subheader("Sample Images from EuroSAT Dataset")
    sample_dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        img, label = sample_dataset[i]
        axes[i//5, i%5].imshow(img.permute(1, 2, 0))
        axes[i//5, i%5].set_title(class_labels[label])
        axes[i//5, i%5].axis('off')
    st.pyplot(fig)
