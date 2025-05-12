import streamlit as st
import os
import zipfile
import requests
import shutil
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image

st.set_page_config(layout="wide")
st.title("EuroSAT Land Use Classification - Streamlit App")

# Set up data directory
data_dir = "data/EuroSAT"

# Download and extract dataset if not exists
@st.cache_data

def download_dataset():
    if not os.path.exists("data/EuroSAT"): 
        os.makedirs("data", exist_ok=True)
        url = "https://drive.google.com/file/d/1NiurPYhckTUhVzzIo4hje7DtaAkzCd2B/view?usp=drive_link"
        r = requests.get(url, stream=True)
        with open("data/eurosat.zip", 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        with zipfile.ZipFile("data/eurosat.zip", 'r') as zip_ref:
            zip_ref.extractall("data")
    return "data/EuroSAT"

data_dir = download_dataset()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes

st.sidebar.header("Dataset Info")
st.sidebar.write(f"Total images: {len(dataset)}")
st.sidebar.write(f"Classes: {class_names}")

# Visualize class distribution
if st.sidebar.checkbox("Show class distribution"):
    class_counts = [0]*len(class_names)
    for _, label in dataset:
        class_counts[label] += 1
    fig1, ax1 = plt.subplots()
    sns.barplot(x=class_names, y=class_counts, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# Visualize random samples per class
if st.sidebar.checkbox("Show image samples"):
    fig2, axs = plt.subplots(len(class_names), 5, figsize=(10, 15))
    for class_idx in range(len(class_names)):
        imgs_shown = 0
        for img, label in dataset:
            if label == class_idx:
                axs[class_idx, imgs_shown].imshow(img.permute(1, 2, 0))
                axs[class_idx, imgs_shown].axis('off')
                imgs_shown += 1
                if imgs_shown == 5:
                    break
        axs[class_idx, 0].set_ylabel(class_names[class_idx], rotation=0, labelpad=40)
    plt.tight_layout()
    st.pyplot(fig2)

# Prepare data
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = SimpleCNN(num_classes=len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train model
if st.button("Train Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_accuracies = []

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        test_accuracies.append(accuracy)

        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy*100:.2f}%")

    # Plot loss and accuracy
    fig3, ax3 = plt.subplots()
    ax3.plot(train_losses, label='Train Loss')
    ax3.plot(test_accuracies, label='Test Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Value')
    ax3.set_title('Training Progress')
    ax3.legend()
    st.pyplot(fig3)

    # Evaluate model
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax4)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig4)

    report = classification_report(all_labels, all_preds, target_names=class_names)
    st.text("Classification Report:\n" + report)

    torch.save(model.state_dict(), "eurosat_model.pth")
    st.success("Model trained and saved as eurosat_model.pth")

