# MindFlow Audio Checkpoints

The trained deep learning model weights (>100MB each) are managed via release artifacts:
- stage1_best.pt (~1.26GB): Fine-tuned WavLM Large emotion backbone (65.8% Val Acc across 7 emotions on 33,397 clips).
- stage2_stress_best.pt (~1.26GB): Fine-tuned continuous stress & PHQ-8 depression regression head (r = 0.389, 71.4% binary accuracy).

To download the pretrained checkpoints, contact the audio branch lead or download from the project drive link.
