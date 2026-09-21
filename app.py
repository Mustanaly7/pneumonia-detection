import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to detect the presence of pneumonia.")

hide_icons_css = """
<style>
/* Hides the top-right menu and cloud deployment icons */
header {
    visibility: hidden;
}
#MainMenu {
    visibility: hidden;
}
</style>
"""
st.markdown(hide_icons_css, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    # Loading the highest accuracy model you just trained
    model.load_state_dict(torch.load('pneumonia_model_best.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_container_width=True)    

    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        prediction = torch.argmax(probabilities).item()
    
    classes = ['Normal', 'Pneumonia']
    confidence = probabilities[prediction].item() * 100
    
    st.subheader(f"Prediction: {classes[prediction]}")
    st.write(f"Confidence: {confidence:.2f}%")