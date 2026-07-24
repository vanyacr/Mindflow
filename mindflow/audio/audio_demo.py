import time
from mic_capture import record_audio, play_audio
from audio_pipeline import predict_emotion

def main():
    print("[START] Real-Time Audio Emotion Detection Started...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # record for 5 seconds now instead of 3
            file = record_audio("temp.wav", duration=5)

            # playback what was just recorded
            play_audio(file)

            print("[*] Processing...")

            # predict
            emotion, state = predict_emotion(file)

            print("\nDetected Emotion:", emotion)
            print("Mental State:", state)
            print("\n-----------------------------")
            
            # Adding a 2-second pause before the next cycle starts
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n[STOP] Stopped Audio Emotion Detection.")

if __name__ == "__main__":
    main()