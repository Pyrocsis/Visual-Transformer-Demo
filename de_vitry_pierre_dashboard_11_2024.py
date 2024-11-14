import streamlit as st
import numpy as np
import os
from PIL import Image, ImageFilter, ImageOps
from transformers import ViTForImageClassification
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import DeiTForImageClassification, DeiTFeatureExtractor, DeiTConfig
import pandas as pd
import torch.nn as nn
from PIL import Image
import cv2
from sklearn.utils.class_weight import compute_class_weight
from captum.attr import Saliency
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import AdamW
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# Set up sidebar for navigation
st.sidebar.title("Menu de Navigation")
st.sidebar.markdown("Utilisez le menu ci-dessous pour naviguer dans l'application.")
page = st.sidebar.selectbox("Aller à la section", ["Accueil", "Analyse Exploratoire", "Visualisation d'Images", "Prédiction", "Accessibilité"])

# # Define paths and model
# img_folder = 'C:/Users/pdevi/OneDrive/Desktop/OpenClassrooms/Projet_6/Images_rgb'
# img_analysis_path = 'C:/Users/pdevi/OneDrive/Desktop/OpenClassrooms/Projet_6/image_analysis.png'
# model_path = 'C:/Users/pdevi/OneDrive/Desktop/OpenClassrooms/Projet_6/best_deit_model.pth'

# Define paths and model
img_folder = 'Images_rgb_sample'
img_analysis_path = 'image_analysis.png'
model_path = 'best_deit_model.pth'
dog_races = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese']
num_classes = len(dog_races)

# Define model creation function for DeiT
def create_deit_model_default(num_classes):
    # Load DeiT with pre-trained weights
    model = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-224',
        num_labels=num_classes, output_hidden_states=True,

    )

    # Freeze all encoder layers initially
    for name, param in model.named_parameters():
        if 'deit' in name:
            param.requires_grad = False

    # Add a new classification head if desired
    model.classifier = nn.Sequential(
        nn.Linear(model.config.hidden_size, 224),  # Hidden layer with reduced size
        nn.ReLU(),
        nn.Dropout(0.5),                           # Dropout for regularization
        nn.Linear(224, num_classes)                # Final classification layer
    )


    return model
num_classes=len(dog_races)

deit_model=create_deit_model_default(num_classes)

deit_model.load_state_dict(torch.load(model_path))
deit_model.eval()

# Home Page
if page == "Accueil":
    st.title("Dashboard de Classification des Races de Chiens")
    st.write("Bienvenue dans le tableau de bord de classification des races de chiens. Ce tableau de bord vous permet d’explorer le jeu de données, de visualiser des images par catégorie, et de réaliser des prédictions en utilisant un modèle Vision Transformer.")
    st.image("C:/Users/pdevi/OneDrive/Desktop/OpenClassrooms/Projet_6/differentes-races-chien.jpg", use_column_width=True)

# Exploratory Data Analysis (EDA) section
elif page == "Analyse Exploratoire":
    st.header("Analyse Exploratoire des Données")
    st.image(img_analysis_path, use_column_width=True)
    
# Image Visualization
elif page == "Visualisation d'Images":
    st.header("Visualisation d'Images avec Transformations")
    
    # Dropdown to select a dog race and reset image index when race changes
    race_index = st.selectbox("Sélectionnez une race de chien", range(len(dog_races)), format_func=lambda x: dog_races[x], key="race_selector")
    selected_race = dog_races[race_index]
    race_folder = os.path.join(img_folder, selected_race)
    
    # Get unique identifiers and reset image index when identifier changes
    all_images = os.listdir(race_folder)
    unique_identifiers = list({img.split('_')[0] + "_" + img.split('_')[1] for img in all_images})
    image_identifier = st.selectbox("Sélectionnez un identifiant d'image", unique_identifiers, key="identifier_selector")
    
    # Reset to the first image (original) every time a new race or identifier is selected
    if st.session_state.get("race_selector") != st.session_state.get("previous_race") or st.session_state.get("identifier_selector") != st.session_state.get("previous_identifier"):
        st.session_state.current_image_index = 0
        st.session_state.previous_race = st.session_state.race_selector
        st.session_state.previous_identifier = st.session_state.identifier_selector

    # Ensure the original image is displayed first
    images = sorted(
        [img for img in os.listdir(race_folder) if img.startswith(image_identifier + "_" and not img.endswith("Constrated.rgb"))],
        key=lambda x: ('Original' not in x, x)
    )

    current_image_name = images[st.session_state.current_image_index]
    current_image_path = os.path.join(race_folder, current_image_name)
    transformation_name = current_image_name.split('_')[2].split('.')[0]
    st.image(current_image_path, caption=f"Transformation: {transformation_name}", use_column_width=True)

    # Navigation buttons to switch images
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Précédent"):
            st.session_state.current_image_index = (st.session_state.current_image_index - 1) % len(images)
    with col3:
        if st.button("Suivant →"):
            st.session_state.current_image_index = (st.session_state.current_image_index + 1) % len(images)

# Prediction Section
elif page == "Prédiction":
    st.header("Prédiction de la Race de Chien")
    uploaded_image = st.file_uploader("Chargez une image de chien pour la prédiction", type=["jpg", "jpeg", "png"])
    
    # Prediction function
    def predict_breed(image):
        img = Image.open(image).resize((224, 224))
        img_array = np.array(img) / 255.0
        print(img_array.shape)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.tensor(img_array).unsqueeze(0)
        with torch.no_grad():
            predictions = deit_model(img_tensor).logits
        predicted_label = dog_races[torch.argmax(predictions).item()]
        confidence = torch.softmax(predictions, dim=1).max().item()
       
        return predicted_label, confidence

    # Display prediction results
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Image Chargée", use_column_width=True)
        predicted_label, confidence = predict_breed(uploaded_image)
        st.write(f"**Race Prédite :** {predicted_label}")
        st.write(f"**Confiance de Prédiction :** {confidence:.2f}")

# Accessibility Section
elif page == "Accessibilité":
    st.header("Accessibilité")
    st.write("Ce tableau de bord prend en compte les critères d'accessibilité pour les utilisateurs en situation de handicap.")
    st.markdown("- Les images sont accompagnées de descriptions textuelles pour les lecteurs d'écran.")
    st.markdown("- Des contrastes élevés et des légendes claires sont utilisés dans les graphiques pour une meilleure lisibilité.")
    st.markdown("- Les éléments interactifs sont faciles d'accès et d'utilisation.")
