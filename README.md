 # Tamil Sign Language Translator (Real-Time)

This project is a real-time Tamil Sign Language Translator that uses hand gesture recognition with MediaPipe and a custom-trained CNN model to detect sign language gestures and convert them into Tamil text and speech.

## Features

- Real-time webcam-based gesture recognition.
- MediaPipe-based hand detection and professional landmark rendering.
- Tamil translation and voice output using Google Text-to-Speech.
- High-accuracy gesture recognition using CNN model. 
- Supports repeated letters and smooth sentence formation.
- Clean and professional UI overlay. 

## Project Structure

Realtime-SignRecognition/
├── main.py                     # Main real-time recognition script
├── model/
│   └── sign_cnn_model.h5       # Trained CNN model for gesture recognition
├── gesture_labels.txt          # Labels file mapping model outputs to characters
├── tamil_translator.py         # Script to translate English to Tamil
├── voice_output.py             # Tamil text-to-speech using gTTS
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)


## Requirements

- Python 3.7+
- OpenCV
- NumPy
- TensorFlow
- MediaPipe
- gTTS
- playsound

# Install requirements using:
```bash
pip install -r requirements.txt
```

# Controls
Press s: Speak the current sentence in Tamil.

Press c: Clear the current sentence.

Press Esc: Exit the application.
