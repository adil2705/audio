from transformers import AutoModelForAudioClassification
import torch
import numpy as np
import io
from pydub import AudioSegment
import streamlit as st

# Load the model
model = AutoModelForAudioClassification.from_pretrained("motheecreator/Deepfake-audio-detection")

# Streamlit app
st.title("Deepfake Audio Detection")

st.write("Upload an audio file to check if it's a deepfake.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Read the audio file
    audio_bytes = uploaded_file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000)  # Ensure the frame rate is compatible
    
    # Convert audio to numpy array
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Ensure audio_data is of shape (1, n_samples) for the model
    audio_data = np.expand_dims(audio_data, axis=0)
    
    # Process the audio (dummy processor if not available)
    inputs = torch.tensor(audio_data)
    
    # Predict
    with torch.no_grad():
        logits = model(inputs).logits
        predicted_class = logits.argmax(-1).item()
    
    # Map the class index to label (if applicable)
    labels = model.config.id2label if hasattr(model.config, 'id2label') else {0: 'unknown'}
    predicted_label = labels.get(predicted_class, 'unknown')
    
    st.write(f"Prediction: {predicted_label}")

