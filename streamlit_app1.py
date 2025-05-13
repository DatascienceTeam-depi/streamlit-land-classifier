import os
import zipfile
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import gdown

# ------------------ 1. Download dataset ------------------
@st.cache_resource
def download_dataset():
    file_id = "1NiurPYhckTUhVzzIo4hje7DtaAkzCd2B"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "dataset.zip"
    extract_to = "data"
    if not os.path.exists(extract_to):
        st.info("Downloading dataset...")
        gdown.download(url, output, quiet=False)
        st.info("Extracting...")
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(output)
        st.success("Dataset ready.")
    else:
        st.info("Dataset already exists.")

# ------------------ 2. Download model ------------------
@st.cache_resource
def download_model():
    model_path = "model.pth"
    if not os.path.exists(model_path):
        file_id = "1VtxQwcYy_-fPccAVfIQI6VQDEdXhwkol"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return model_path

# ------------------ 3. Inspect folder structure ------------------
@st.cache_data
def inspect_folder_structure(path="data"):
    tree = []
    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = "  " * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        for f in files:
            tree.append(f"{indent}  {f}")
    return tree

# ------------------ 4. Load model correctly ------------------
@st.cache_resource
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ------------------ 5. Main App ------------------
def main():
    st.title("🌍 Land Type Classifier")
    
    download_dataset()
    model_path = download_model()
    
    # Show dataset structure
    st.subheader("📁 Dataset Structure")
    for line in inspect_folder_structure():
        st.text(line)

    # Prepare class names
    classes = sorted([d.name for d in os.scandir('data') if d.is_dir()])

    # Load model
    model = load_model(model_path, num_classes=len(classes))

    # Sidebar input
    st.sidebar.header("Upload an image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Transform and predict
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = classes[preds.item()]

        st.subheader("Prediction")
        st.success(f"**{predicted_class}**")

if __name__ == "__main__":
    main()
