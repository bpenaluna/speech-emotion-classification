import numpy as np
from flask import Flask, request, render_template
import torch
import torchvision
import torch.nn.functional as F
from python_speech_features import mfcc
import librosa
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = torch.jit.load('speech-classifier.pth', map_location=torch.device('cpu'))
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# load audio
def load_audio(audio_path):
  signal, rate = librosa.load(audio_path)
  signal = librosa.effects.preemphasis(signal, coef=0.97) # apply preemphasis filter
  signal = librosa.effects.trim(signal, top_db=40)[0] # trim leading and trailing silence
  signal = librosa.util.fix_length(signal, size=57344)
  return signal, rate

def predict_emotion(file_path):
    classes = ['NEUTRAL', 'CALM', 'HAPPY', 'SAD', 'ANGRY', 'FEARFUL', 'DISGUST', 'SURPRISED']

    print('Loading audio...')
    signal, rate = librosa.load(file_path)
    print('Preprocessing audio...')
    signal = librosa.effects.preemphasis(signal, coef=0.97) # apply preemphasis filter
    signal = librosa.util.fix_length(signal, size=57344)
    input = mfcc(signal)
    input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
    print('Predicting...')
    model.eval()
    with torch.no_grad():
      prediction = model(input).argmax(dim=1)
    print("Prediction:", classes[prediction.item()])

    return classes[prediction]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['filename']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    prediction = predict_emotion(file_path)
    os.remove(os.path.join(UPLOAD_FOLDER, file.filename))

    return "Predicted emotion: " + prediction

if __name__ == '__main__':
    app.run(debug=True)