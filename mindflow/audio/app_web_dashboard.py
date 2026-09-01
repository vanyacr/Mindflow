"""
MindFlow — Interactive Visual Web Dashboard (Gradio & Plotly)
=============================================================
A presentation-ready web application for evaluators and reviewers.

Features:
1. Live Microphone Recording & Audio File Upload.
2. Interactive Emotion Distribution Bar & Radar Charts.
3. Speedometer / Gauge for PHQ-8 Continuous Stress Score.
4. Personalized User Voice Baseline Calibration Tab.
5. Multimodal Fusion Vector (768-D) Inspector.

Launch with:
    python app_web_dashboard.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import UNIFIED_EMOTIONS
from inference.audio_interface import AudioInference
from inference.user_calibration import UserProfile, UserProfileCalibrator

# Load Audio Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Web Dashboard] Initializing MindFlow Audio Model on {device.upper()}...")
model = AudioInference(device=device)
print("[Web Dashboard] Model loaded successfully!")

EMOTION_COLORS = {
    "happy": "#2ecc71",
    "sad": "#3498db",
    "angry": "#e74c3c",
    "fear": "#9b59b6",
    "neutral": "#95a5a6",
    "surprise": "#f39c12",
    "disgust": "#16a085",
}

# Global active profile
active_user_profile: UserProfile | None = None


def plot_emotion_bars(emotion_probs: dict[str, float]) -> go.Figure:
    sorted_items = sorted(emotion_probs.items(), key=lambda x: x[1])
    emotions = [k.capitalize() for k, _ in sorted_items]
    probs = [v * 100 for _, v in sorted_items]
    colors = [EMOTION_COLORS.get(k.lower(), "#34495e") for k, _ in sorted_items]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=emotions,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.2)", width=1)),
            text=[f"{p:.1f}%" for p in probs],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="<b>Emotion Probability Distribution (7 Classes)</b>",
        xaxis=dict(title="Probability (%)", range=[0, 100]),
        yaxis=dict(title="Emotion"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        template="plotly_white",
    )
    return fig


def plot_stress_gauge(stress_val: float) -> go.Figure:
    score_pct = stress_val * 100
    if stress_val < 0.25:
        severity = "Minimal / Normal"
        bar_color = "#2ecc71"
    elif stress_val < 0.50:
        severity = "Mild Stress"
        bar_color = "#f1c40f"
    elif stress_val < 0.75:
        severity = "Moderate Stress (Elevated)"
        bar_color = "#e67e22"
    else:
        severity = "Severe / High Stress"
        bar_color = "#e74c3c"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score_pct,
            title={"text": f"<b>PHQ-8 Continuous Stress Score</b><br><span style='font-size:0.8em;color:gray'>{severity}</span>"},
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 25], "color": "rgba(46, 204, 113, 0.25)"},
                    {"range": [25, 50], "color": "rgba(241, 196, 15, 0.25)"},
                    {"range": [50, 75], "color": "rgba(230, 126, 34, 0.25)"},
                    {"range": [75, 100], "color": "rgba(231, 76, 60, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 41.7,  # PHQ-8 >= 10 threshold (10/24)
                },
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), template="plotly_white")
    return fig


def analyze_audio(audio_path: str, use_calibration: bool):
    if not audio_path:
        return "Please record or upload an audio file first.", None, None, "", ""

    global active_user_profile
    prof = active_user_profile if use_calibration else None

    res = model.predict(audio_path, user_profile=prof)

    top_emo = res.get("calibrated_emotion", res["emotion"]).upper()
    probs = res.get("calibrated_emotion_probs", res["emotion_probs"])
    top_prob = probs[res.get("calibrated_emotion", res["emotion"])] * 100
    stress = res.get("calibrated_stress", res["stress"])

    fig_bars = plot_emotion_bars(probs)
    fig_gauge = plot_stress_gauge(stress)

    primary_text = f"## 🎯 Primary Emotion: **{top_emo}** ({top_prob:.1f}% Confidence)\n### 📊 Stress Index: **{stress:.3f}** (Continuous PHQ-8)"

    # Acoustic delta metadata
    meta_text = "### 🔬 Personalized Voice Biomarkers:\n"
    if "calibration_metadata" in res:
        m = res["calibration_metadata"]
        meta_text += f"- **Pitch ($F_0$) Shift**: `{m['pitch_delta_pct']:+.1f}%`\n"
        meta_text += f"- **Energy Delta**: `{m['energy_delta_db']:+.1f} dB`\n"
        meta_text += f"- **Pause Ratio Shift**: `{m['pause_ratio_delta']:+.2f}`\n"
        meta_text += f"- **Acoustic Cosine Sim**: `{m['cosine_similarity_to_base']:.3f}`\n"
        if m["clinical_markers"]:
            meta_text += f"- ⚠️ **Clinical Triggers**: `{', '.join(m['clinical_markers'])}`\n"
        else:
            meta_text += "- ✅ **Clinical Triggers**: `Normal vocal range`\n"
    else:
        meta_text += "*No baseline calibrated. Using population model.*"

    # Multimodal Vector Summary
    emb = res["embedding"]
    emb_preview = f"**768-D Fusion Vector (First 6 dims)**: `{[round(x, 4) for x in emb[:6]]}`\n**L2 Norm**: `{np.linalg.norm(emb):.3f}`"

    return primary_text, fig_bars, fig_gauge, meta_text, emb_preview


def calibrate_profile(calib_audio: str, user_name: str):
    if not calib_audio:
        return "Please record neutral calibration speech first."

    global active_user_profile
    user_id = user_name.strip() or "demo_user"
    active_user_profile = model.register_user_baseline(calib_audio, user_id=user_id)

    msg = f"### ✅ Baseline Registered for User: **'{user_id}'**\n"
    msg += f"- **Fundamental Pitch ($F_0$)**: `{active_user_profile.base_pitch_mean:.1f} Hz`\n"
    msg += f"- **Pitch Std Dev**: `{active_user_profile.base_pitch_std:.1f} Hz`\n"
    msg += f"- **Base Energy (RMS)**: `{active_user_profile.base_energy_rms:.4f}`\n"
    msg += f"- **Base Pause Rate**: `{active_user_profile.base_pause_ratio * 100:.1f}%`\n"
    msg += f"- Profile saved to `profiles/{user_id}_profile.json`"
    return msg


# Build Gradio Blocks UI
with gr.Blocks(title="MindFlow — Audio Emotion & Stress AI") as demo:
    gr.Markdown(
        """
        # 🧠 **MindFlow — Speech Emotion & Clinical Stress Intelligence**
        ### Multimodal Audio Branch powered by **Fine-Tuned WavLM Large + Self-Attention Pooling**
        """
    )

    with gr.Tabs():
        with gr.TabItem("🎙️ Live Emotion & Stress Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record from Microphone or Upload WAV File",
                    )
                    use_calib_checkbox = gr.Checkbox(
                        label="Apply Personalized User Voice Calibration",
                        value=True,
                    )
                    analyze_btn = gr.Button("🚀 Run Real-Time Audio Analysis", variant="primary", size="lg")

                with gr.Column(scale=2):
                    primary_output = gr.Markdown("### Click 'Run Real-Time Audio Analysis' to evaluate speech.")
                    with gr.Row():
                        plot_emotions = gr.Plot(label="Emotion Probabilities")
                        plot_gauge = gr.Plot(label="PHQ-8 Continuous Stress Score")

            with gr.Row():
                with gr.Column():
                    biomarker_output = gr.Markdown("")
                with gr.Column():
                    vector_output = gr.Markdown("")

            analyze_btn.click(
                fn=analyze_audio,
                inputs=[audio_input, use_calib_checkbox],
                outputs=[primary_output, plot_emotions, plot_gauge, biomarker_output, vector_output],
            )

        with gr.TabItem("👤 Voice Profile Calibration"):
            gr.Markdown(
                """
                ### **Personalized Voice Baseline Calibration**
                Record **6 to 8 seconds** of speech in a calm, neutral tone.
                MindFlow will compute your baseline $F_0$ pitch and volume to eliminate soft-spoken or mic volume bias.
                """
            )
            with gr.Row():
                with gr.Column():
                    user_name_input = gr.Textbox(label="User / Patient ID", value="patient_001")
                    calib_audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Neutral Voice Sample")
                    calib_btn = gr.Button("💾 Register & Save Baseline Profile", variant="secondary")
                with gr.Column():
                    calib_status = gr.Markdown("### Awaiting calibration recording...")

            calib_btn.click(
                fn=calibrate_profile,
                inputs=[calib_audio_input, user_name_input],
                outputs=[calib_status],
            )

        with gr.TabItem("ℹ️ Model Architecture & Technical Specs"):
            gr.Markdown(
                """
                ### 🏗️ **Technical Pipeline Summary**
                * **Backbone**: `WavLM Large` (316M Parameters) — Layers 1–12 frozen, Layers 13–24 fine-tuned on 33,397 clips.
                * **Pooling**: Learnable Self-Attention Temporal Pooling ($h_t \rightarrow c$).
                * **Stage 1 (Emotion)**: 7-Class Weighted Cross-Entropy Loss on 6 Datasets (**Macro-F1: 0.631, Accuracy: 65.8%**).
                * **Stage 2 (Clinical Stress)**: 4-Layer MLP on 266 DAIC-WOZ Clinical Interviews (**Pearson $r = 0.389$, Binary Acc: 71.4%**).
                * **Multimodal Vector**: Standardized **768-D embedding** passed to Multimodal Late Fusion layer.
                """
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="127.0.0.1", server_port=7860, share=False)
