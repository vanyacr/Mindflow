"""Demo script: Keystroke dynamics for stress detection."""

from __future__ import annotations

import time

from KEYSTROKE import run_keystroke_pipeline
from KEYSTROKE.keystroke_listener import KeystrokeListener


def demo_keystroke_baseline():
    """Demo 1: Normal typing baseline (low stress)."""
    print("\n" + "="*70)
    print("DEMO 1: Normal Typing Baseline (Low Stress)")
    print("="*70)
    
    listener = KeystrokeListener()
    listener.set_user_baseline_wpm(60.0)
    buffer = listener.get_buffer()
    
    # Simulate normal typing: consistent speed, short holds, minimal pauses
    print("Simulating: User typing a casual message at normal pace...")
    current_time = time.time()
    
    # Simulate 30 keystrokes with normal timing (~60 WPM relative to 5 chars/word)
    for i in range(30):
        # Press event
        listener.inject_event(
            key=f"char_{i}",
            event_type="press",
            timestamp=current_time + (i * 0.2),  # ~0.2s between keypresses
        )
        # Release event (hold ~80ms)
        listener.inject_event(
            key=f"char_{i}",
            event_type="release",
            timestamp=current_time + (i * 0.1) + 0.08,
            duration=0.08,
        )
    
    result = run_keystroke_pipeline(keystroke_buffer=buffer, user_baseline_wpm=60.0)
    
    print(f"Keystroke Stress Probability: {result['stress_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"WPM Component: {result['component_scores']['wpm']['score']:.2%} ({result['component_scores']['wpm']['reason']})")
    print(f"Pressure Component: {result['component_scores']['pressure']['score']:.2%} ({result['component_scores']['pressure']['reason']})")
    print(f"Pause Component: {result['component_scores']['pauses']['score']:.2%} ({result['component_scores']['pauses']['reason']})")
    print(f"Events captured: {result['event_count']}")
    
    return result


def demo_keystroke_stressed():
    """Demo 2: Stressed typing (high deviation, long holds, frequent pauses)."""
    print("\n" + "="*70)
    print("DEMO 2: Stressed Typing (High Stress)")
    print("="*70)
    
    listener = KeystrokeListener()
    listener.set_user_baseline_wpm(60.0)
    buffer = listener.get_buffer()
    
    print("Simulating: User typing under pressure (hesitations, slower pace, harder presses)...")
    current_time = time.time()
    
    # Simulate stressed typing: slower, longer holds, long pauses
    event_idx = 0
    for group in range(4):  # 4 word-like groups
        start_time = current_time + (group * 3.0)  # 3 seconds between groups (hesitation)
        
        for i in range(5):  # 5 keystrokes per "word"
            press_time = start_time + (i * 0.2)  # slower: 0.2s between keys
            
            listener.inject_event(
                key=f"char_{event_idx}",
                event_type="press",
                timestamp=press_time,
            )
            # Stressed: longer hold (150ms) + occasional very long holds
            hold_duration = 0.15 if i % 2 == 0 else 0.25
            listener.inject_event(
                key=f"char_{event_idx}",
                event_type="release",
                timestamp=press_time + hold_duration,
                duration=hold_duration,
            )
            event_idx += 1
    
    result = run_keystroke_pipeline(keystroke_buffer=buffer, user_baseline_wpm=60.0)
    
    print(f"Keystroke Stress Probability: {result['stress_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"WPM Component: {result['component_scores']['wpm']['score']:.2%} ({result['component_scores']['wpm']['reason']})")
    print(f"Pressure Component: {result['component_scores']['pressure']['score']:.2%} ({result['component_scores']['pressure']['reason']})")
    print(f"Pause Component: {result['component_scores']['pauses']['score']:.2%} ({result['component_scores']['pauses']['reason']})")
    print(f"Events captured: {result['event_count']}")
    
    return result


def demo_keystroke_excited():
    """Demo 3: Excited/rushed typing (elevated WPM, short holds, minimal pauses)."""
    print("\n" + "="*70)
    print("DEMO 3: Excited/Rushed Typing (Elevated Arousal)")
    print("="*70)
    
    listener = KeystrokeListener()
    listener.set_user_baseline_wpm(60.0)
    buffer = listener.get_buffer()
    
    print("Simulating: User typing rapidly (excited or in a hurry)...")
    current_time = time.time()
    
    # Simulate fast typing: rapid keystrokes, short holds
    for i in range(50):
        # Very short intervals: 0.05s (simulating ~1200 WPM equivalent)
        listener.inject_event(
            key=f"char_{i}",
            event_type="press",
            timestamp=current_time + (i * 0.05),
        )
        # Short hold (40ms)
        listener.inject_event(
            key=f"char_{i}",
            event_type="release",
            timestamp=current_time + (i * 0.05) + 0.04,
            duration=0.04,
        )
    
    result = run_keystroke_pipeline(keystroke_buffer=buffer, user_baseline_wpm=60.0)
    
    print(f"Keystroke Arousal Probability: {result['stress_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"WPM Component: {result['component_scores']['wpm']['score']:.2%} ({result['component_scores']['wpm']['reason']})")
    print(f"Pressure Component: {result['component_scores']['pressure']['score']:.2%} ({result['component_scores']['pressure']['reason']})")
    print(f"Pause Component: {result['component_scores']['pauses']['score']:.2%} ({result['component_scores']['pauses']['reason']})")
    print(f"Events captured: {result['event_count']}")
    
    return result


def main():
    """Run all keystroke demos."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " " * 15 + "KEYSTROKE DYNAMICS STRESS DETECTION DEMO" + " " * 13 + "║")
    print("╚" + "="*68 + "╝")
    
    baseline_result = demo_keystroke_baseline()
    stressed_result = demo_keystroke_stressed()
    excited_result = demo_keystroke_excited()
    
    print("\n" + "="*70)
    print("SUMMARY: Keystroke Modality Scores")
    print("="*70)
    
    print(f"\nBaseline (Normal):    {baseline_result['stress_probability']:.2%} stress")
    print(f"Stressed (High):      {stressed_result['stress_probability']:.2%} stress  ← detected hesitation")
    print(f"Excited (Rapid):      {excited_result['stress_probability']:.2%} stress  ← high WPM deviation")
    
    print("\n✅ Keystroke module successfully integrated as 4th modality!")
    print("📊 Ready for fusion layer integration with TEXT, AUDIO, VIDEO.")
    print()


if __name__ == "__main__":
    main()
