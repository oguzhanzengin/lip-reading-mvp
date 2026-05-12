# Turkish Lip Reading MVP

A real-time lip reading prototype built with Python, OpenCV, MediaPipe and TensorFlow.

This project detects the mouth region from webcam input, collects short mouth movement sequences, trains a sequence-based deep learning model, and performs real-time word prediction.

## Current Features

- Real-time webcam capture
- Face and mouth region detection with MediaPipe Face Mesh
- Custom data collection pipeline
- Image sequence preprocessing
- CNN + LSTM based word classification
- Real-time prediction from mouth movement

## Current Classes

The current MVP recognizes 3 Turkish words:

- evet
- hayir
- tamam

## Project Structure

```text
lip-reading-mvp/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── detect_mouth.py
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── realtime_predict.py
├── requirements.txt
├── README.md
└── .gitignore


Installation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Usage
1. Detect mouth region
python src/detect_mouth.py
2. Collect training data
python src/collect_data.py
Controls:
SPACE = record sequence
N = next word
Q = quit
3. Preprocess dataset
python src/preprocess.py
4. Train model
python src/train_model.py
5. Run real-time prediction
python src/realtime_predict.py
Controls:
SPACE = test one word
Q = quit
Model
The current model uses:
TimeDistributed CNN
→ LSTM
→ Dense classifier
Input shape:
20 frames × 50 height × 100 width × 3 channels
Current Limitations
This is an early MVP trained on a very small personal dataset.
Accuracy is limited and may vary depending on lighting, camera angle, distance and speaking speed.
Roadmap
Stage 1 — Personal MVP
Webcam-based data collection
Mouth crop extraction
CNN + LSTM model
Real-time word prediction
Stage 2 — Turkish Lip Reading Dataset
Integrate an existing Turkish visual lip reading dataset
Convert external data into the same sequence format
Retrain and compare model performance
Stage 3 — Pretrained AVSR Model
Test pretrained audio-visual speech recognition models
Evaluate inference quality on sample videos
Stage 4 — YouTube Data Pipeline
Download Turkish speech videos
Extract subtitles
Detect and crop speaker mouth region
Align video segments with transcript text
Stage 5 — Turkish Fine-Tuning
Fine-tune a pretrained visual/audio-visual speech recognition model
Improve Turkish sentence-level recognition
Long-Term Goal
The long-term goal is to build a Turkish visual speech recognition system that can understand spoken words or short phrases from mouth movement, and eventually combine visual and audio signals for more robust speech recognition.