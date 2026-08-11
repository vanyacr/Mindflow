import os
# Suppress TensorFlow boot logs and warnings to keep terminal clean!
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from audio_features import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model_dl.h5")

# Safely check if DL model exists
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[SUCCESS] Loaded Deep Learning CNN-LSTM model successfully!")
else:
    print("[WARNING] 'emotion_model_dl.h5' not found! You must train and download the model first.")

REVERSE_EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad"
}

def map_to_state(emotion):
    if emotion in ["sad", "fear", "disgust"]:
        return "stressed"
    elif emotion == "angry":
        return "high_stress"
    elif emotion == "neutral":
        return "normal"
    elif emotion == "happy":
        return "relaxed"
    else:
        return "unknown"

def predict_emotion(file_path):
    if model is None:
        return "MODEL MISSING", "UNAVAILABLE"

    # Deep learning extraction yields 2D array: (150, 180)
    features = extract_features(file_path)
    
    # Model requires a batch dimension so we expand it: (1, 150, 180)
    features = np.expand_dims(features, axis=0)

    # Output probabilities and selecthighest
    predictions = model.predict(features, verbose=0)
    class_index = np.argmax(predictions[0])
    
    emotion = REVERSE_EMOTIONS[class_index]
    state = map_to_state(emotion)

    return emotion, state