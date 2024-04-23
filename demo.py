import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import glob
import os

import librosa
import librosa.display

import torch
from torch import nn
from torchvision import models, transforms, datasets

from time import time
from tqdm import tqdm

from PIL import Image

import streamlit as st

seed = 12
np.random.seed(seed)

path = "../input/gtzan-dataset-music-genre-classification/"

path_audio_files = path + "Data/genres_original/"

path_imgs = "./mel_spectrogram_imgs/"

batch_size = 32

hop_length = 512

n_fft = 2048

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

genre_dict = {"blues":0,"classical":1,"country":2,"disco":3,"hiphop":4,"jazz":5,"metal":6,"pop":7,"reggae":8,"rock":9}
class_dict = {0: "blues", 1: "classical", 2: "country", 3: "disco", 4: "hiphop", 
                  5: "jazz", 6: "metal", 7: "pop", 8: "reggae", 9: "rock"}


def create_mel_spectrogram_image(audio_file_path, image_save_path, sampling_rate=22050, hop_length=512):
    # Load the audio file
    # In the GTZAN dataset, each song is 30 s long, with a 22,050 Hz sample rate, 
    # mono mode, AU file format, and 16-bit audio files
    data, sr = librosa.load(audio_file_path, sr=sampling_rate)

    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, hop_length=hop_length)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot and save the mel spectrogram as an image
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length)
    plt.axis('off')  # Optional: Remove axes for a cleaner image
    plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def classify_audio(image_path, model, num_classes=10):    
    img = Image.open(image_path).convert('RGB')
    img_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4931, 0.9151, 0.9960], std=[0.4495, 0.1716, 0.0602])
    ])
    
    img_normalized = img_transformer(img)
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)

    # Predict with both models
    with torch.no_grad():
        model.eval()
        prediction = model(img_normalized)

    # Convert predictions to probabilities
    probs = nn.functional.softmax(prediction, dim=1)

    # Get the top 3 predictions and their indices
    top_probs, top_indices = torch.topk(probs, 10)

    # Convert the top probabilities to percentages
    top_percentages = [round(prob.item() * 100, 2) for prob in top_probs[0]]
    
    # Convert indices to class names
    top_genres = [class_dict[index.item()] for index in top_indices[0]]

    # Combine class names with their corresponding percentages
    predictions = list(zip(top_genres, top_percentages))

    return predictions

resnet = torch.load("resnet_best",map_location=torch.device('cpu'))

st.title('Music Genre Prediction')
st.write('Upload a music file to visualize its mel spectrogram and predict its genre.')

music_file = st.file_uploader("Choose a music file", type=['mp3'])

if music_file is not None:
    # Save uploaded file to a temporary file to process
    with open(os.path.join("temp_music_file.mp3"), "wb") as f:
        f.write(music_file.getbuffer())
        
    # Create and display mel spectrogram
    create_mel_spectrogram_image("temp_music_file.mp3", "mel_spec.png")
    spectrogram_image = Image.open("mel_spec.png").convert('RGB')
    st.image(spectrogram_image, caption='Mel Spectrogram', use_column_width=True)
    
    # Predict and display genre
    genre_prediction = classify_audio("mel_spec.png",resnet)
    st.write(f'Output: {genre_prediction}')
else:
    st.write("Please upload a file to get started.")


