# MindFlow Audio Branch — 4 Core Presentation Slides (Slide Deck Text)

---

## 📽️ SLIDE 1: Problem Formulation & Model Architecture

### **Title**: MindFlow Audio Branch — Deep Speech Emotion & Clinical Stress Intelligence
#### **Subtitle**: Leveraging Foundation WavLM Large & Self-Attention Pooling for Multimodal Mental Health Assessment

### **Key Bullet Points**:
* **The Core Challenge**: Human speech exhibits extreme variability in acoustic prosody, speaking speed, pitch, and mic background noise across individuals.
* **Foundation Backbone**: `WavLM Large` (316M parameters, 24 Transformer layers) pre-trained on 94,000 hours of masked speech denoising.
* **Targeted Layer Unfreezing Strategy**:
  * **Layers 1–12 (Frozen)**: Preserves foundational phonetic and acoustic feature representations.
  * **Layers 13–24 (Fine-Tuned)**: Actively fine-tuned to capture emotional prosody, cadence, and vocal tension.
* **Learnable Self-Attention Pooling**: Dynamically scores and weights emotional vocal bursts ($\alpha_t$) rather than taking a naive uniform mean across silent pauses.
* **Multimodal Late Fusion Vector**: Linearly projects WavLM embeddings ($1024 \rightarrow 768$) to output a dense 768-D acoustic representation that feeds into the downstream Video + Text Multimodal Fusion Engine.

### **Speaker Notes / Talking Points**:
> *"Good morning. In our MindFlow Audio Branch, we built an end-to-end speech processing pipeline that solves two clinical tasks simultaneously: classifying 7 discrete emotion states and quantifying continuous depression severity on the PHQ-8 scale. Rather than training from scratch, we used Microsoft's WavLM Large foundation model, fine-tuned the top 12 layers on over 33,000 speech clips, and used a learned self-attention pooling layer to output a 768-dimensional multimodal fusion vector."*

---

## 📽️ SLIDE 2: Dataset Breadth & Rigorous Speaker-Disjoint Split

### **Title**: Multi-Corpus Training Strategy & Generalization Rigor
#### **Subtitle**: 33,397 Diverse Audio Clips across 6 Benchmark Corpora + DAIC-WOZ Clinical Sessions

### **Key Bullet Points**:
* **Comprehensive 6-Dataset Training Corpus (33,397 Clips)**:
  * **Acted Benchmarks**: CREMA-D (7,442 clips), RAVDESS (1,440 clips), SAVEE (480 clips), TESS (2,800 clips).
  * **In-The-Wild Conversational Benchmarks**: MELD (13,706 clips from multi-speaker dialogue), IEMOCAP (7,529 spontaneous interactive turns).
* **Strict Speaker-Disjoint Validation**:
  * Evaluated on a 100% speaker-independent validation set (2,887 clips).
  * The model is tested *strictly on unseen speakers* it has never heard during training to prevent identity memorization.
* **Class Imbalance Resolution**:
  * Utilized **Inverse-Frequency Class Weighting** ($w_c = \frac{N}{C \cdot N_c}$) during backpropagation to prevent majority classes from dominating minority emotions (e.g. Surprise, Disgust).

### **Speaker Notes / Talking Points**:
> *"A major pitfall in academic speech emotion recognition is evaluating on the same speakers used during training, which leads to artificial 90% accuracy that collapses in the real world. We enforced a strict Speaker-Disjoint split across 33,397 clips. We also integrated in-the-wild datasets like MELD and IEMOCAP so the model handles background noise, conversational interruptions, and real spontaneous speech."*

---

## 📽️ SLIDE 3: Quantitative Results & Clinical Benchmark Validation

### **Title**: Empirical Results & Performance Validation
#### **Subtitle**: Stage 1 Emotion Recognition & Stage 2 Clinical Stress Regression

### **Key Bullet Points**:
* **Stage 1 (7-Class Speech Emotion Recognition)**:
  * **Overall Accuracy**: **65.8%** on unseen speakers (**~4.6× higher than random chance of 14.3%**).
  * **Macro-F1 Score**: **0.631** (Happy: 0.679, Angry: 0.764, Neutral: 0.663, Fear: 0.605, Disgust: 0.610, Sad: 0.578).
  * **Clinical Context**: Matches human inter-annotator agreement on raw audio (60–70%), serving as the optimal feature extractor for late multimodal fusion.
* **Stage 2 (DAIC-WOZ Continuous Stress / PHQ-8 Regression)**:
  * 5-Fold Stratified Cross-Validation on 266 clinical diagnostic interview sessions.
  * **Pearson Correlation ($r$)**: **$0.389 \pm 0.067$** (up from baseline $\sim 0.185$).
  * **Binary Depression Classification Accuracy**: **$71.4\% \pm 5.6\%$** ($\text{PHQ-8} \ge 10$ clinical threshold).
  * **Mean Absolute Error (MAE)**: **$0.205 \pm 0.012$** on normalized $[0, 1]$ scale.

### **Speaker Notes / Talking Points**:
> *"Our Stage 1 model achieves 65.8% accuracy and 0.631 Macro-F1 across 7 classes on completely unseen speakers. In literature, human listeners only agree 60 to 70% of the time when hearing audio without facial cues. On Stage 2, evaluated on the clinical DAIC-WOZ dataset, our continuous stress head achieved a Pearson correlation of 0.389 and 71.4% binary accuracy in identifying moderate-to-severe depression thresholds."*

---

## 📽️ SLIDE 4: Real-Time Personalized Calibration & Live Streaming Innovation

### **Title**: Personalized Acoustic Calibration & Real-Time Streaming
#### **Subtitle**: Overcoming Individual Speaker Bias & Live Sliding-Window Monitoring

### **Key Bullet Points**:
* **The Personalization Problem**: Naturally soft-spoken or high-pitched speakers are often falsely flagged as depressed or anxious by static population models.
* **Onboarding Baseline Calibration (6–8 Seconds)**:
  * Extracts user's neutral Fundamental Pitch ($F_0$ via YIN autocorrelation), baseline RMS energy, and pause ratio.
  * Calculates relative acoustic deltas ($\Delta F_0\%$, $\Delta\text{dB}$, Cosine Embedding Drift).
* **Automated Clinical Biomarker Detection**:
  * *Monotone / Flat Affect*: Detected when $\Delta F_0 \le -25\%$ and $\Delta\text{dB} \le -3\text{dB}$.
  * *Psychomotor Slowing / High Latency*: Detected when Pause Ratio Delta $\ge +0.25$.
* **Continuous Sliding-Window Live Streaming Engine**:
  * Uses a rolling 6.0-second circular buffer updated every 1.0 second for continuous real-time monitoring.
  * Interactive Web UI (Gradio & Plotly) for live microphone testing and visual emotion gauges.

### **Speaker Notes / Talking Points**:
> *"Our key innovation is the Personalized Voice Baseline Calibration. In just 6 seconds of neutral speech, MindFlow profiles the user's pitch, energy, and speech latency. When analyzing subsequent speech, it measures relative shifts—such as a 30% pitch drop or reduced energy indicating psychomotor slowing—rather than applying rigid population thresholds. We have implemented this in a live sliding-window web dashboard that continuously analyzes microphone input every second."*
