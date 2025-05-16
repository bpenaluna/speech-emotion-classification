import numpy as np
from flask import Flask, request, render_template
import torch
import torchvision
import torch.nn.functional as F
from python_speech_features import mfcc
import librosa
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = torch.jit.load('speech-classifier.pth', map_location=torch.device('cpu'))

# load audio
def load_audio(audio_path):
  signal, rate = librosa.load(audio_path)
  signal = librosa.effects.preemphasis(signal, coef=0.97) # apply preemphasis filter
  signal = librosa.util.fix_length(signal, size=95646)
  return signal, rate

# preprocessing
def preprocess(signal, rate):
  MFCC = mfcc(signal)

  # convert to mel spectogram
  #mel_signal = librosa.feature.melspectrogram(y=signal, sr=rate, hop_length=512, n_fft=1024, n_mels=60)
  #mel_signal_db = librosa.power_to_db(np.abs(mel_signal), ref=np.max) # map magnitudes to decibel scale

  # normalise the spectogram
  #mel_signal -= mel_signal.min()
  #mel_signal /= mel_signal.max()
  #mel_signal = (mel_signal * 255).astype(np.uint8)

  # convert to colour image with shape (60, 172, 3)
  #rgb_img = cv2.applyColorMap(mel_signal, cv2.COLORMAP_JET)
  #rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert to RGB

  # resize
  #tensor = torch.tensor(rgb_img).permute(2, 0, 1).float() / 255.0 # shape (60, 172, 3) --> (3, 60, 172)
  #tensor = tensor.unsqueeze(0) # add dimension: (3, 60, 172) --> (1, 3, 60, 172)
  #tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False) # reshape to (1, 3, 224, 224)

  return torch.tensor(MFCC, dtype=torch.float32)

def predict_emotion(file_path):
    classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    print('Loading audio...')
    signal, rate = librosa.load(file_path)
    signal = librosa.effects.preemphasis(signal, coef=0.97) # apply preemphasis filter
    signal = librosa.util.fix_length(signal, size=95646)
    input = preprocess(signal, rate).unsqueeze(0)
    print(input.shape)
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