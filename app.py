import tkinter as tk
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import tensorflow as tf
import pickle
import os

# Load the trained model
model = tf.keras.models.load_model("C:/Users/gourab/Desktop/Speech_emotion_recogniser/audio_sentiment_model.h5")

# Load the label encoder
with open("C:/Users/gourab/Desktop/Speech_emotion_recogniser/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Audio recording function (6 seconds)
def record_audio(filename="recorded.wav", duration=6, fs=44100):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    write(filename, fs, recording)

# Feature extraction using MFCC
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Predict emotion and update GUI
def predict_emotion():
    status_label.config(text="🎙️ Recording... Please speak now.")
    root.update()

    record_audio()
    features = extract_features("recorded.wav")
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])

    status_label.config(text=f"🧠 Predicted Emotion: {predicted_label[0].capitalize()}")

# GUI setup
root = tk.Tk()
root.title("Speech Emotion Recognizer")
root.geometry("400x250")
root.configure(bg="#dfe6e9")

title = tk.Label(root, text="🎙️ Speech Emotion Recognizer", font=("Helvetica", 16, "bold"), bg="#dfe6e9")
title.pack(pady=20)

record_btn = tk.Button(root, text="Record and Predict", font=("Helvetica", 12), command=predict_emotion, bg="#74b9ff", fg="white", padx=10, pady=5)
record_btn.pack(pady=10)

status_label = tk.Label(root, text="Click the button and speak for 6 seconds", font=("Helvetica", 12), bg="#dfe6e9")
status_label.pack(pady=20)

root.mainloop()
