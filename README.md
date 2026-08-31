# 🧠 MindFlow — AI-Driven Mental Well-Being & Productivity Companion

**Final Year Capstone Project**
*Real-Time Multimodal Intelligence • AI Fusion Engine • Gamified Self-Regulation Framework*

---

## 🚀 Project Overview

MindFlow is a real-time, end-to-end **multimodal AI system** that monitors, analyzes, and enhances student well-being and productivity during study sessions.

The system integrates three parallel data streams:

| Modality | Signal |
|---|---|
| 📷 **Video** | Facial behavior, engagement, cognitive signals |
| 🎙️ **Audio** | Speech emotion, vocal stress |
| ⌨️ **Text** | Sentiment, psychological indicators |

These streams are fused inside a core **AI Brain (Multimodal Fusion Engine)** to generate unified metrics:

- 🎯 **Focus Percentage**
- ⚡ **Productivity Score**
- 🔥 **Burnout Risk**
- 💚 **Well-Being Index**

The system concludes with a **real-time analytics dashboard**, a **gamification & motivation engine**, and **long-term progress tracking**.

---

## 🏗️ System Architecture — 9-Stage Pipeline

### 1️⃣ Student Session Initiation
- Student opens the application
- Webcam + microphone activated, text input enabled
- Session timer begins

### 2️⃣ Multimodal Input Layer
Simultaneous capture of:
- 📷 Video stream
- 🎙️ Audio waveform
- ⌨️ Text input

### 3️⃣ Video Processing — Computer Vision Pipeline
**Techniques used:**
- Face detection (MediaPipe)
- Emotion recognition (trained on AffectNet)
- Engagement estimation (DAiSEE)
- Cognitive load detection (CogLoad)
- Eye gaze tracking, head pose estimation, blink & drowsiness detection

**Outputs:** Engagement Level · Stress Probability · Distraction Score · Fatigue Index · Cognitive Load

### 4️⃣ Audio Processing — Speech & Vocal Analysis
**Feature extraction:** MFCC · Pitch & Energy · Spectral Features
**Trained on:** RAVDESS · CREMA-D

**Outputs:** Speech Emotion · Vocal Stress Score · Energy Level

### 5️⃣ Text Processing — NLP Pipeline
**Models & datasets:** BERT-based sentiment analysis · Psychological indicators from DAIC-WOZ

**Outputs:** Sentiment Polarity · Anxiety Probability · Emotional Tone · Motivation Level

### 6️⃣ Multimodal Fusion Engine — *Core AI Brain* 🧠
The most critical module of the system.

**Fusion strategy:**
- Feature-level aggregation
- Confidence-weighted scoring
- Temporal smoothing
- Context-aware recalibration

**Generates:** Focus Percentage · Productivity Score · Burnout Risk · Well-Being Index · Time-Series Stability Metrics

This layer converts raw behavioral signals into actionable intelligence.

### 7️⃣ Analytics Dashboard
Real-time visualization layer displaying:
- Focus % gauge
- Engagement meter
- Mood trend graph
- Productivity timeline
- Cognitive load indicator
- Stress alerts

### 8️⃣ Gamification & Feedback Engine
Designed to promote **self-regulation, not surveillance.**
- 🔥 Focus streak counter
- 🎁 Reward points
- 🎯 Goal tracking
- ☕ Smart break suggestions
- 💬 Motivational notifications

### 9️⃣ Data Logging & Progress Tracking
Long-term behavioral analytics:
- 📋 Session summary
- 📅 Weekly trends
- 📆 Monthly reports
- 📈 Improvement analysis

---

## 🛠️ Technology Stack

**Programming & Frameworks**
`Python` · `OpenCV` · `MediaPipe` · `TensorFlow` / `PyTorch` · `HuggingFace Transformers` · `Scikit-learn` · `Flask` / `FastAPI` · `Streamlit` / `React`

**Core Domains**
Computer Vision · Speech Processing · Natural Language Processing · Multimodal Deep Learning · Human-Centered AI

---

## 🎓 Research Contribution

**Proposed research title:**
> *"Multimodal AI-Based Student Well-Being and Productivity Monitoring: Design, Implementation, and Evaluation"*

**Research gap addressed:**
- ✔ Lack of unified real-time multimodal monitoring frameworks
- ✔ No integration of AI analytics + gamification
- ✔ Limited tools for student self-regulation
- ✔ Absence of adaptive behavioral intelligence systems

### 📊 Research Novelty Analysis
Detailed comparison available in [`/novelty`](./novelty).

We benchmark against **15+ state-of-the-art models** across:
- Video modality
- Audio modality
- Text modality
- Multimodal fusion

**Our contribution:**
- Real-time unified multimodal system
- Student-centric gamification
- Integrated productivity + well-being metrics

---

## 🔒 Ethical & Privacy Considerations

- Consent-based monitoring
- Local processing where possible
- No third-party data sharing
- Student-first design philosophy
- Built as a **support system, not a surveillance tool**

---

## 📁 Project Structure

```
├── data/
├── models/
│   ├── video_models/
│   ├── audio_models/
│   ├── text_models/
├── fusion_engine/
├── dashboard/
├── gamification/
├── backend/
├── utils/
└── README.md
```

---

## 🌟 Key Highlights

- ✅ 9-stage end-to-end workflow
- ✅ Real-time multimodal fusion
- ✅ AI-driven unified well-being index
- ✅ Gamified motivation framework
- ✅ Research & publication ready
- ✅ Scalable modular architecture

---

## 👩‍💻 Authors

| Name | Role |
|---|---|
| **Vanya C R** | Video Module |
| **Srujana T** | Fusion Engine, Dashboard, Gamification |
| **Vaishnavi Mudhole** | Audio Module |
| **Sathwik S K** | Text / NLP Module |

B.Tech CSE, PES University, Karnataka
