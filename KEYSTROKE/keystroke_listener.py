"""Keystroke event capture and buffer management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class KeystrokeEvent:
    """Single keystroke event with timing and metadata."""

    key: str  # key name
    event_type: str  # 'press' or 'release'
    timestamp: float  # absolute time (seconds since epoch)
    duration: float = 0.0  # for release events: key hold duration


@dataclass
class KeystrokeBuffer:
    """Rolling buffer of keystroke events within a session."""

    events: List[KeystrokeEvent] = field(default_factory=list)
    session_start_time: float = field(default_factory=time.time)
    user_baseline_wpm: float = 60.0  # default: assume 60 WPM baseline
    
    def add_event(self, event: KeystrokeEvent) -> None:
        """Add a keystroke event to the buffer."""
        self.events.append(event)
    
    def get_session_duration(self) -> float:
        """Return elapsed time since session start (seconds)."""
        return time.time() - self.session_start_time
    
    def get_events_in_window(self, window_seconds: int = 60) -> List[KeystrokeEvent]:
        """Get keystroke events from the last N seconds."""
        cutoff_time = time.time() - window_seconds
        return [e for e in self.events if e.timestamp >= cutoff_time]
    
    def clear_old_events(self, keep_seconds: int = 600) -> None:
        """Remove keystroke events older than the retention window (default 10 min)."""
        cutoff_time = time.time() - keep_seconds
        self.events = [e for e in self.events if e.timestamp >= cutoff_time]
    
    def reset_session(self) -> None:
        """Clear buffer and reset session timer."""
        self.events.clear()
        self.session_start_time = time.time()


class KeystrokeListener:
    """Mock keyboard listener for development (can be replaced with pynput/pyxhook in production)."""
    
    def __init__(self, buffer: KeystrokeBuffer | None = None):
        """Initialize listener with optional pre-configured buffer."""
        self.buffer = buffer or KeystrokeBuffer()
        self.is_listening = False
        self.listener_handle = None  # placeholder for actual listener hook
    
    def start(self) -> None:
        """Start listening to keystroke events (mock: ready for real implementation)."""
        self.is_listening = True
        print("[keystroke_listener] Listening started (mock mode)")
    
    def stop(self) -> None:
        """Stop listening to keystroke events."""
        self.is_listening = False
        if self.listener_handle:
            self.listener_handle.stop()
        print("[keystroke_listener] Listening stopped")
    
    def inject_event(self, key: str, event_type: str, timestamp: float | None = None, duration: float = 0.0) -> None:
        """For testing: directly inject a keystroke event."""
        if timestamp is None:
            timestamp = time.time()
        event = KeystrokeEvent(key=key, event_type=event_type, timestamp=timestamp, duration=duration)
        self.buffer.add_event(event)
    
    def get_buffer(self) -> KeystrokeBuffer:
        """Return the underlying keystroke buffer."""
        return self.buffer
    
    def set_user_baseline_wpm(self, wpm: float) -> None:
        """Set the user's typical typing speed for WPM deviation calculation."""
        self.buffer.user_baseline_wpm = max(20.0, min(150.0, float(wpm)))
