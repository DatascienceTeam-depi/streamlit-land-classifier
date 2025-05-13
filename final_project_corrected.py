import os
import zipfile
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import gdown

# 1. Download and extract dataset from Google Drive
@st.cache_resource
def download_dataset_from_gdrive():
    file_id = "1NiurPYhckTUhVzzIo4hje7DtaAkzCd2B"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "dataset.zip"
    extract_to = "data"

    if not os.path.exists(extract_to):
        st.info("Downloading dataset from Google Drive...")
        gdown.download(url, output, quiet=False)
        st.info("Extracting dataset...")
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(output)
        st.success("Dataset is ready!")
    else:
        st.info("Dataset already exists.")

# 1.5 Download model from Google Drive
@st.cache_resource
def download_model_from_gdrive():
    model_path = "model.pth"
    if not os.path.exists(model_path):
        file_id = "1VtxQwcYy_-fPccAVfIQI6VQDEdXhwkol"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

# 2. Inspect folder structure
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

# 3. Option 1: Load model from state_dict
def load_model_option_1(model_path="model.pth", num_classes=6):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        st.success("Model loaded successfully (Option 1).")
    else:
        st.error(f"Model file '{model_path}' not found.")
    model.eval()
    return model

# 3. Option 2: Load full model directly
def load_model_option_2(model_path="model.pth"):
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=torch.device('cpu'))
        st.success("Model loaded successfully (Option 2).")
        model.eval()
        return model
    else:
        st.error(f"Model file '{model_path}' not found.")
        return None

# 4. Main Streamlit App
def main():
    st.title("🌍 Land Type Classifier")

    # Download and prepare data and model
    download_dataset_from_gdrive()
    download_model_from_gdrive()

    # Show folder tree
    st.subheader("📁 Dataset Folder Structure")
    structure = inspect_folder_structure()
    for line in structure:
        st.text(line)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Prepare classes from folder names
    classes = sorted([d.name for d in os.scandir('data') if d.is_dir()])

    # Choose loading option
    load_option = st.sidebar.selectbox("Model Load Option", ["Option 1 (state_dict)", "Option 2 (full model)"])
    if load_option == "Option 1 (state_dict)":
        model = load_model_option_1(model_path="model.pth", num_classes=len(classes))
    else:
        model = load_model_option_2(model_path="model.pth")

    st.sidebar.header("🔧 Options")
    uploaded_file = st.sidebar.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = classes[preds.item()] if preds.item() < len(classes) else "Unknown"

        st.subheader("🖼️ Prediction")
        st.write(f"**Predicted Class:** {predicted_class}")

if __name__ == "__main__":
    main()
