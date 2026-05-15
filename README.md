# Speech Emotion Classifier

A Flask web application that classifies human emotions from audio recordings using a pre-trained PyTorch deep learning model. Upload a `.wav` file and get back one of eight predicted emotion labels.

## Emotions

The model can classify the following emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## How It Works

1. An audio file is uploaded via the web interface.
2. The audio is preprocessed using `librosa` — a pre-emphasis filter is applied, silence is trimmed, and the signal is padded/truncated to a fixed length.
3. Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from the signal using `python_speech_features`.
4. The MFCC features are passed to a quantized TorchScript model (`speech-classifier.pth`) for inference.
5. The predicted emotion label is returned and displayed.

## Project Structure

```
speech-emotion-classification/
├── classifier.py          # Flask app — audio preprocessing, inference, and routes
├── speech-classifier.pth  # Pre-trained TorchScript model
├── requirements.txt       # Python dependencies
├── static/                # Static assets (CSS, JS)
└── templates/             # HTML templates
    └── index.html         # Upload form
```

## Installation

**Prerequisites:** Python 3.8+

1. Clone the repository:
   ```bash
   git clone https://github.com/bpenaluna/speech-emotion-classification.git
   cd speech-emotion-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** `librosa` requires `ffmpeg` for some audio formats. Install it via your system package manager (e.g. `brew install ffmpeg` on macOS or `apt install ffmpeg` on Ubuntu).

## Usage

1. Start the Flask development server:
   ```bash
   python classifier.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000`.

3. Upload an audio file using the form and click submit — the predicted emotion will be displayed on the page.

## Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web framework |
| `torch` / `torchvision` | Model loading and inference |
| `librosa` | Audio loading and preprocessing |
| `python_speech_features` | MFCC feature extraction |
| `numpy` | Numerical operations |
| `opencv-python` | Image/array utilities |

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Model

The classifier is loaded from `speech-classifier.pth`, a TorchScript model that is dynamically quantized at runtime (8-bit integer quantization on linear layers) to reduce memory usage and improve CPU inference speed.
