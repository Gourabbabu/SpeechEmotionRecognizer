# 🎙️ Speech Emotion Recognizer

A simple GUI-based Python app that records your voice and detects emotions using a trained deep learning model (using MFCC features and a TensorFlow model).

---

## 💡 Features
- 🎤 Record live voice for 6 seconds
- 🤖 Predict emotion using a pre-trained model
- 🖥️ User-friendly GUI with `tkinter`
- 📊 Real-time audio processing and prediction

---

## 🛠️ Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
SpeechEmotionRecognizer/
├── main.py                 # Main GUI logic
├── audio_sentiment_model.h5 # Trained emotion detection model
├── encoder.pkl             # Label encoder for emotion classes
├── recorded.wav            # Temporary audio file (ignored by .gitignore)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Usage

```bash
python main.py
```

1. Click "Record and Predict".
2. Speak for 6 seconds.
3. View the predicted emotion on screen.

---

## 🧠 Model Details

- Trained on **RAVDESS dataset**
- Uses **MFCC features** for audio signal processing
- Built with **TensorFlow/Keras**
- Label encoding via **scikit-learn**

---

## ✅ How to Clone and Run

1. **Clone the repo**

```bash
git clone https://github.com/Gourabbabu/SpeechEmotionRecognizer.git
cd SpeechEmotionRecognizer
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
python main.py
```

---

## 📁 Notes

- `.wav` files and virtual environments are excluded via `.gitignore`
- Make sure your mic is enabled and accessible to Python
- This app is tested on **Windows**, may require small tweaks for **Linux/macOS**

---

## 🧑‍💻 Author

**Gourab Chakraborty**  
[GitHub](https://github.com/Gourabbabu)

---

Let me know if you'd like a logo, screenshots section, or contribution instructions too!
