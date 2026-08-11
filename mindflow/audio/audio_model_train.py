import os
import zipfile
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from audio_features import extract_features
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = "dataset/"

EMOTIONS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

def extract_and_organize_zips():
    print("Checking for Datasets...")
    os.makedirs("dataset", exist_ok=True)
    for emo in EMOTIONS.keys():
        os.makedirs(f"dataset/{emo}", exist_ok=True)

    if os.path.exists("Ravdess.zip") and not os.path.exists("ravdess_temp"):
        print("Unzipping Ravdess.zip...")
        with zipfile.ZipFile("Ravdess.zip", 'r') as zip_ref:
            zip_ref.extractall("ravdess_temp")
            
        ravdess_map = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fear", "07": "disgust"}
        for root, dirs, files in os.walk("ravdess_temp"):
            for file in files:
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if len(parts) > 2 and parts[2] in ravdess_map:
                        emo = ravdess_map[parts[2]]
                        shutil.move(os.path.join(root, file), f"dataset/{emo}/ravdess_{file}")
        print("Organized Ravdess successfully.")

    if os.path.exists("Crema-D.zip") and not os.path.exists("crema_temp"):
        print("Unzipping Crema-D.zip...")
        with zipfile.ZipFile("Crema-D.zip", 'r') as zip_ref:
            zip_ref.extractall("crema_temp")

        crema_map = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
        for root, dirs, files in os.walk("crema_temp"):
            for file in files:
                if file.endswith(".wav"):
                    parts = file.split("_")
                    if len(parts) >= 3 and parts[2] in crema_map:
                        emo = crema_map[parts[2]]
                        shutil.move(os.path.join(root, file), f"dataset/{emo}/crema_{file}")
        print("Organized Crema-D successfully.")

def load_data():
    X = []
    y = []

    print("\nPre-processing all audio (This will take time locally)...")
    for emotion_dir in os.listdir(DATASET_PATH):
        if emotion_dir.lower() in EMOTIONS:
            folder_path = os.path.join(DATASET_PATH, emotion_dir)
            files = os.listdir(folder_path)
            for i, file_name in enumerate(files):
                if file_name.endswith('.wav') and not file_name.startswith("._"):
                    file_path = os.path.join(folder_path, file_name)
                    
                    try:
                        features = extract_features(file_path) # shape: (150, 180)
                        X.append(features)
                        y.append(EMOTIONS[emotion_dir.lower()])
                    except Exception as e:
                        print(f"  -> Skipping corrupted/hidden file {file_name}")
                        continue
                    
                    if len(X) > 0 and len(X) % 100 == 0:
                        print(f"  -> Successfully extracted {len(X)} files so far...")
                        
    return np.array(X), np.array(y)

def build_model(input_shape, num_classes):
    model = Sequential()
    
    # Built-in normalization so we don't need scaler.pkl!
    model.add(BatchNormalization(input_shape=input_shape))

    # CNN Layer to extract local audio patterns (cracks in voice, sharp frequencies)
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # LSTM Layer to track emotion over the flow of time
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    
    # Classification head
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    extract_and_organize_zips()

    X, y = load_data()
    
    if len(X) == 0:
        print("\n[ERROR] No data found.")
        print("Please drag your 'Ravdess.zip' and 'Crema-D.zip' directly into this 'audio' folder in VS Code, then run this script again!")
        exit()
        
    y_cat = to_categorical(y, num_classes=len(EMOTIONS))
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    
    print(f"\nTraining data shape: {X_train.shape}")
    
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=len(EMOTIONS))
    model.summary()
    
    print("\nStarting Deep Learning Training (This will heavily utilize your CPU/GPU)...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    print("\nEvaluating Model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    model.save("emotion_model_dl.h5")
    print("\n✅ Saved deep learning model to 'emotion_model_dl.h5'!")
