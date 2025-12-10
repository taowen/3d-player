#!/usr/bin/env python3
"""
Player ViewModel
Pure business logic and state management without Qt dependencies
"""

from reaktiv import Signal, Computed


class PlayerViewModel:
    """
    ViewModel for video player - contains all business logic and reactive state
    Completely decoupled from Qt for easy testing
    """

    def __init__(self):
        # === Reactive State (Signals) ===
        # Playback state
        self.position = Signal(0)  # Current playback position in ms
        self.duration = Signal(0)  # Total duration in ms
        self.is_playing = Signal(False)  # Playing state
        self.is_dragging = Signal(False)  # User dragging slider

        # UI state
        self.volume = Signal(50)  # Volume 0-100
        self.is_fullscreen = Signal(False)  # Fullscreen state
        self.file_path = Signal("")  # Current file path

        # === Computed Values (Derived State) ===
        self.progress_percent = Computed(lambda:
            (self.position() / self.duration() * 1000) if self.duration() > 0 else 0
        )

        self.current_time_text = Computed(lambda:
            self.format_time(self.position())
        )

        self.duration_time_text = Computed(lambda:
            self.format_time(self.duration())
        )

        self.file_name = Computed(lambda:
            self._get_file_name(self.file_path())
        )

        self.is_file_loaded = Computed(lambda:
            bool(self.file_path())
        )

        self.file_display_text = Computed(lambda:
            f"Playing: {self.file_name()}" if self.is_file_loaded() else "No file loaded"
        )

        # For icon type determination (returns string identifiers)
        self.play_icon_type = Computed(lambda:
            "pause" if self.is_playing() else "play"
        )

        self.fullscreen_icon_type = Computed(lambda:
            "restore" if self.is_fullscreen() else "maximize"
        )

    # === State Update Methods ===

    def set_position(self, position_ms: int):
        """Update playback position"""
        self.position.set(position_ms)

    def set_duration(self, duration_ms: int):
        """Update total duration"""
        self.duration.set(duration_ms)

    def set_playing_state(self, is_playing: bool):
        """Update playing state"""
        self.is_playing.set(is_playing)

    def set_volume(self, volume: int):
        """Update volume (0-100)"""
        if 0 <= volume <= 100:
            self.volume.set(volume)

    def set_dragging(self, is_dragging: bool):
        """Update dragging state"""
        self.is_dragging.set(is_dragging)

    def set_fullscreen(self, is_fullscreen: bool):
        """Update fullscreen state"""
        self.is_fullscreen.set(is_fullscreen)

    def set_file_path(self, file_path: str):
        """Update current file path"""
        self.file_path.set(file_path)

    # === Action Methods ===

    def toggle_play(self) -> bool:
        """
        Toggle play/pause state
        Returns: True if should play, False if should pause
        """
        new_state = not self.is_playing()
        self.is_playing.set(new_state)
        return new_state

    def toggle_fullscreen(self):
        """Toggle fullscreen state"""
        self.is_fullscreen.set(not self.is_fullscreen())

    def start_dragging(self):
        """Start slider dragging"""
        self.is_dragging.set(True)

    def stop_dragging(self):
        """Stop slider dragging"""
        self.is_dragging.set(False)

    def calculate_position_from_slider(self, slider_value: int, slider_max: int = 1000) -> int:
        """
        Calculate media position from slider value

        Args:
            slider_value: Current slider value (0-slider_max)
            slider_max: Maximum slider value (default 1000)

        Returns:
            Position in milliseconds
        """
        dur = self.duration()
        if dur > 0 and slider_max > 0:
            return int((slider_value / slider_max) * dur)
        return 0

    def get_slider_value_from_position(self, slider_max: int = 1000) -> int:
        """
        Calculate slider value from current position

        Args:
            slider_max: Maximum slider value (default 1000)

        Returns:
            Slider value (0-slider_max)
        """
        return int(self.progress_percent() * slider_max / 1000)

    def get_volume_ratio(self) -> float:
        """Get volume as ratio (0.0-1.0) for audio output"""
        return self.volume() / 100.0

    def handle_play_action(self) -> str:
        """
        Determine what play action should be taken

        Returns:
            "play" or "pause" action
        """
        return "pause" if self.is_playing() else "play"

    def handle_stop_action(self):
        """Handle stop action - reset playing state"""
        self.is_playing.set(False)

    def handle_file_loaded(self, file_path: str):
        """Handle when a new file is loaded"""
        self.file_path.set(file_path)
        # Auto-play on file load
        self.is_playing.set(True)

    def handle_escape_key(self):
        """Handle escape key - exit fullscreen if active"""
        if self.is_fullscreen():
            self.is_fullscreen.set(False)

    def handle_f11_key(self):
        """Handle F11 key - toggle fullscreen"""
        self.toggle_fullscreen()

    def handle_space_key(self) -> bool:
        """
        Handle space key - toggle play/pause

        Returns:
            True if should play, False if should pause
        """
        return self.toggle_play()

    # === Utility Methods ===

    @staticmethod
    def format_time(milliseconds: int) -> str:
        """
        Format time in milliseconds to HH:MM:SS

        Args:
            milliseconds: Time in milliseconds

        Returns:
            Formatted time string (HH:MM:SS)
        """
        if milliseconds < 0:
            return "00:00:00"

        seconds = milliseconds // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _get_file_name(file_path: str) -> str:
        """
        Extract file name from path

        Args:
            file_path: Full file path

        Returns:
            File name or empty string
        """
        if not file_path:
            return ""

        import os
        return os.path.basename(file_path)

    # === Query Methods ===

    def should_update_slider(self) -> bool:
        """Check if slider should be updated (not while dragging)"""
        return not self.is_dragging()

    def is_ready_for_playback(self) -> bool:
        """Check if a file is loaded and ready for playback"""
        return self.is_file_loaded() and self.duration() > 0

    def get_progress_info(self) -> dict:
        """
        Get complete progress information

        Returns:
            Dict with position, duration, progress_percent, and formatted times
        """
        return {
            "position_ms": self.position(),
            "duration_ms": self.duration(),
            "progress_percent": self.progress_percent(),
            "current_time": self.current_time_text(),
            "duration_time": self.duration_time_text(),
        }

    def get_state_snapshot(self) -> dict:
        """
        Get a complete snapshot of the current state
        Useful for debugging and testing

        Returns:
            Dict containing all state values
        """
        return {
            "position": self.position(),
            "duration": self.duration(),
            "is_playing": self.is_playing(),
            "is_dragging": self.is_dragging(),
            "volume": self.volume(),
            "is_fullscreen": self.is_fullscreen(),
            "file_path": self.file_path(),
            "progress_percent": self.progress_percent(),
            "current_time": self.current_time_text(),
            "duration_time": self.duration_time_text(),
            "file_name": self.file_name(),
            "is_file_loaded": self.is_file_loaded(),
        }
