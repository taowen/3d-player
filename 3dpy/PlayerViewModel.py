#!/usr/bin/env python3
"""
Player ViewModel
Pure business logic and state management without Qt dependencies
"""

from reaktiv import Signal, Computed


class PlayerViewModel:
    """ViewModel for video player - business logic and reactive state"""

    def __init__(self) -> None:
        self.position = Signal(0)
        self.duration = Signal(0)
        self.is_playing = Signal(False)
        self.is_dragging = Signal(False)
        self.volume = Signal(50)
        self.is_fullscreen = Signal(False)
        self.file_path = Signal("")

        self.progress_percent = Computed(lambda:
            (self.position() / self.duration() * 1000) if self.duration() > 0 else 0
        )
        self.current_time_text = Computed(lambda: self.format_time(self.position()))
        self.duration_time_text = Computed(lambda: self.format_time(self.duration()))
        self.file_name = Computed(lambda: self._get_file_name(self.file_path()))
        self.is_file_loaded = Computed(lambda: bool(self.file_path()))
        self.file_display_text = Computed(lambda:
            f"Playing: {self.file_name()}" if self.is_file_loaded() else "No file loaded"
        )
        self.play_icon_type = Computed(lambda: "pause" if self.is_playing() else "play")
        self.fullscreen_icon_type = Computed(lambda: "restore" if self.is_fullscreen() else "maximize")

    def set_position(self, position_ms: int) -> None:
        self.position.set(position_ms)

    def set_duration(self, duration_ms: int) -> None:
        self.duration.set(duration_ms)

    def set_playing_state(self, is_playing: bool) -> None:
        self.is_playing.set(is_playing)

    def set_volume(self, volume: int) -> None:
        if 0 <= volume <= 100:
            self.volume.set(volume)

    def set_dragging(self, is_dragging: bool) -> None:
        self.is_dragging.set(is_dragging)

    def set_fullscreen(self, is_fullscreen: bool) -> None:
        self.is_fullscreen.set(is_fullscreen)

    def set_file_path(self, file_path: str) -> None:
        self.file_path.set(file_path)

    def toggle_play(self) -> bool:
        new_state = not self.is_playing()
        self.is_playing.set(new_state)
        return new_state

    def toggle_fullscreen(self) -> None:
        self.is_fullscreen.set(not self.is_fullscreen())

    def start_dragging(self) -> None:
        self.is_dragging.set(True)

    def stop_dragging(self) -> None:
        self.is_dragging.set(False)

    def calculate_position_from_slider(self, slider_value: int, slider_max: int = 1000) -> int:
        dur = self.duration()
        if dur > 0 and slider_max > 0:
            return int((slider_value / slider_max) * dur)
        return 0

    def get_slider_value_from_position(self, slider_max: int = 1000) -> int:
        return int(self.progress_percent() * slider_max / 1000)

    def get_volume_ratio(self) -> float:
        return self.volume() / 100.0

    def handle_stop_action(self) -> None:
        self.is_playing.set(False)

    def handle_file_loaded(self, file_path: str) -> None:
        self.file_path.set(file_path)
        self.is_playing.set(True)

    def handle_escape_key(self) -> None:
        if self.is_fullscreen():
            self.is_fullscreen.set(False)

    def handle_f11_key(self) -> None:
        self.toggle_fullscreen()

    def handle_space_key(self) -> bool:
        return self.toggle_play()

    @staticmethod
    def format_time(milliseconds: int) -> str:
        if milliseconds < 0:
            return "00:00:00"

        seconds = milliseconds // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _get_file_name(file_path: str) -> str:
        if not file_path:
            return ""
        import os
        return os.path.basename(file_path)

    def should_update_slider(self) -> bool:
        return not self.is_dragging()
