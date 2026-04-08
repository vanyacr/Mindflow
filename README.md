<div align="center">
🧠 MindFlow
AI-Powered Mental Wellbeing & Productivity Companion
Real-Time Multimodal Intelligence · AI Fusion Engine · Gamified Self-Regulation

<br/>

Final Year Capstone Project — B.Tech CSE, PES University, Karnataka
Designed as a support system, not a surveillance tool.

</div>

📌 Overview
MindSync is a real-time, end-to-end Multimodal AI system that monitors, analyzes, and enhances student well-being and productivity during study sessions. It fuses three parallel data streams — video, audio, and text — into a unified intelligence layer that produces actionable, student-centric insights.
📷 Video  ──┐
🎙️ Audio  ──┼──▶  🧠 Multimodal Fusion Engine  ──▶  📊 Dashboard + 🏆 Gamification
⌨️  Text   ──┘
Core Output Metrics
MetricDescription🎯 Focus PercentageReal-time attention and engagement level⚡ Productivity ScoreComposite work-efficiency rating🔥 Burnout RiskEarly warning signal for cognitive overload💚 Well-Being IndexHolistic emotional and mental health score

🏗️ System Architecture — 9-Stage Pipeline
┌─────────────────────────────────────────────────────────────────┐
│                    STUDENT SESSION                               │
│  Webcam ON · Microphone ON · Text Input ON · Timer Starts       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   📷 VIDEO STREAM   🎙️ AUDIO WAVEFORM  ⌨️ TEXT INPUT
          │                │                │
          ▼                ▼                ▼
  ┌───────────────┐ ┌────────────────┐ ┌──────────────────┐
  │  CV Pipeline  │ │ Speech & Vocal │ │   NLP Pipeline   │
  │               │ │    Analysis    │ │                  │
  │ · Face Detect │ │ · MFCC         │ │ · BERT Sentiment │
  │ · Emotion Rec │ │ · Pitch/Energy │ │ · Anxiety Detect │
  │ · Gaze Track  │ │ · Spectral     │ │ · Motivation     │
  │ · Drowsiness  │ │   Features     │ │   Scoring        │
  └───────┬───────┘ └───────┬────────┘ └────────┬─────────┘
          │                 │                    │
          └────────┬────────┘────────────────────┘
                   ▼
     ┌─────────────────────────────┐
     │  🧠 MULTIMODAL FUSION ENGINE │
     │                             │
     │  · Feature-level Aggregation│
     │  · Confidence-weighted Score│
     │  · Temporal Smoothing       │
     │  · Context-aware Recalibrate│
     └──────────────┬──────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  📊 ANALYTICS           🏆 GAMIFICATION
  DASHBOARD              ENGINE
         │                     │
         └──────────┬──────────┘
                    ▼
           📂 DATA LOGGING &
           PROGRESS TRACKING

🔬 Processing Pipelines
<details>
<summary><b>📷 Stage 3 — Video Processing (Computer Vision)</b></summary>
<br/>
Techniques Used:

Face Detection via MediaPipe
Emotion Recognition trained on AffectNet
Engagement Estimation via DAiSEE
Cognitive Load Detection via CogLoad
Eye Gaze Tracking & Head Pose Estimation
Blink & Drowsiness Detection

Outputs:
SignalDescriptionEngagement LevelAttention continuity scoreStress ProbabilityVisual cue-based stress estimationDistraction ScoreFocus deviation metricFatigue IndexDrowsiness and blink-rate analysisCognitive LoadMental effort estimation
</details>
<details>
<summary><b>🎙️ Stage 4 — Audio Processing (Speech & Vocal Analysis)</b></summary>
<br/>
Feature Extraction:

MFCC (Mel-Frequency Cepstral Coefficients)
Pitch & Energy Analysis
Spectral Features

Training Datasets: RAVDESS · CREMA-D
Outputs: Speech Emotion · Vocal Stress Score · Energy Level
</details>
<details>
<summary><b>⌨️ Stage 5 — Text Processing (NLP Pipeline)</b></summary>
<br/>
Models & Datasets:

BERT-based Sentiment Analysis
Psychological Indicators from DAIC-WOZ

Outputs: Sentiment Polarity · Anxiety Probability · Emotional Tone · Motivation Level
</details>
<details>
<summary><b>🧠 Stage 6 — Multimodal Fusion Engine (Core AI Brain)</b></summary>
<br/>
The most critical module. Converts raw behavioral signals from all three modalities into unified, actionable intelligence.
Fusion Strategy:

Feature-level aggregation across modalities
Confidence-weighted scoring per stream
Temporal smoothing for stability
Context-aware recalibration

Final Outputs: Focus % · Productivity Score · Burnout Risk · Well-Being Index · Time-Series Stability Metrics
</details>
<details>
<summary><b>📊 Stage 7 — Analytics Dashboard</b></summary>
<br/>
Real-time visualization of:

Focus % Gauge
Engagement Meter
Mood Trend Graph
Productivity Timeline
Cognitive Load Indicator
Stress Alerts

</details>
<details>
<summary><b>🏆 Stage 8 — Gamification & Feedback Engine</b></summary>
<br/>
Designed to promote self-regulation, not surveillance.
FeatureDescription🔥 Focus Streak CounterRewards sustained attention🎁 Reward PointsMilestone-based achievement system🎯 Goal TrackingPersonalized session targets☕ Smart Break SuggestionsContext-aware rest prompts💬 Motivational NotificationsPositive behavioral reinforcement
</details>
<details>
<summary><b>📂 Stage 9 — Data Logging & Progress Tracking</b></summary>
<br/>
Long-term behavioral analytics:

📋 Session Summary
📅 Weekly Trends
📆 Monthly Reports
📈 Improvement Analysis

</details>

🛠️ Technology Stack
LayerTechnologiesCore LanguagePython 3.10+Computer VisionOpenCV, MediaPipeDeep LearningTensorFlow, PyTorchNLPHuggingFace Transformers, BERTML UtilitiesScikit-learnBackendFlask / FastAPIFrontendStreamlit / React

📁 Project Structure
mindsync/
├── data/                      # Raw and processed datasets
├── models/
│   ├── video_models/          # CV & emotion recognition models
│   ├── audio_models/          # Speech emotion models
│   └── text_models/           # BERT-based NLP models
├── fusion_engine/             # 🧠 Core multimodal fusion logic
├── dashboard/                 # Analytics & visualization layer
├── gamification/              # Rewards & motivation engine
├── backend/                   # API endpoints (Flask/FastAPI)
├── novelty/                   # Research novelty analysis (15+ SOTA comparisons)
├── utils/                     # Shared utilities and helpers
└── README.md

🎓 Research Contribution
Proposed Research Title:

"Multimodal AI-Based Student Well-Being and Productivity Monitoring: Design, Implementation, and Evaluation"

Research Gaps Addressed

✅ Lack of unified real-time multimodal monitoring frameworks
✅ No prior integration of AI analytics + gamification for student well-being
✅ Limited tools for student self-regulation
✅ Absence of adaptive behavioral intelligence systems for education

Novelty Analysis
We compare 15+ state-of-the-art models across video, audio, text, and multimodal fusion modalities. Full comparison available in /novelty.
Our system's improvements:

Real-time unified multimodal architecture
Student-centric gamification layer
Integrated productivity + well-being metrics (not studied in isolation)


🔒 Ethics & Privacy
This system is built with a student-first philosophy:

🔐 Consent-based monitoring — explicit opt-in required
💻 Local processing where possible — data stays on-device
🚫 No third-party data sharing
🛡️ Designed as a support and empowerment tool, never surveillance


👩‍💻 Authors
<div align="center">
NameRoleSrujana TCo-AuthorVanya CRCo-AuthorVaishnavi MudholeCo-AuthorSathwik SKCo-Author
B.Tech Computer Science & Engineering
PES University, Bengaluru, Karnataka
</div>

<div align="center">
MindSync — Because your well-being is as important as your GPA.
</div>
