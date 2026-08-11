import librosa
import numpy as np
import warnings
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress librosa warnings for clean terminal
warnings.filterwarnings('ignore', category=UserWarning)

MAX_TIME_STEPS = 150 # Fixed timeline frames roughly equal to 3.5 seconds

def extract_features(file_path):
    # Automatically loads the entire length of whatever was recorded
    audio, sr = librosa.load(file_path)

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=25)
    
    # Normalize Volume
    if len(audio) > 0:
        audio = librosa.util.normalize(audio)
    else:
        audio = np.zeros(sr * 3)

    # DEEP LEARNING UPGRADE:
    # Notice we removed np.mean(). We want the raw array over time!
    # Transpose (.T) so we get shape (time_steps, features)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T
    mel = librosa.feature.melspectrogram(y=audio, sr=sr).T

    # Combine into array of shape (time_steps, 180)
    features = np.hstack((mfcc, chroma, mel))

    # Pad or cut exactly to MAX_TIME_STEPS so the neural network can ingest it properly
    features_padded = pad_sequences([features], maxlen=MAX_TIME_STEPS, dtype='float32', padding='post', truncating='post')

    return features_padded[0]