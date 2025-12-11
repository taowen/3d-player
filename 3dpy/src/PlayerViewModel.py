#!/usr/bin/env python3
"""
Player ViewModel
Pure business logic and state management without Qt dependencies
"""

from dataclasses import dataclass
from typing import List
from reaktiv import Signal, Computed


@dataclass
class SubtitleEntry:
    """Single subtitle entry with timing and text"""
    start_time_ms: int
    end_time_ms: int
    text: str


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
        self.subtitle_file_path = Signal("")

        # Private subtitle data
        self._subtitle_entries: List[SubtitleEntry] = []

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
        self.current_subtitle_text = Computed(lambda: self._get_current_subtitle())
        self.is_subtitle_loaded = Computed(lambda: bool(self.subtitle_file_path()))

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

    # Subtitle methods
    def load_subtitle_file(self, file_path: str) -> bool:
        """Load and parse SRT subtitle file. Returns True on success."""
        try:
            entries = self._parse_srt_file(file_path)
            self._subtitle_entries = entries
            self.subtitle_file_path.set(file_path)
            return True
        except Exception as e:
            print(f"Failed to load subtitle: {e}")
            return False

    def clear_subtitle(self) -> None:
        """Clear current subtitle"""
        self._subtitle_entries = []
        self.subtitle_file_path.set("")

    def _get_current_subtitle(self) -> str:
        """Get subtitle text at current playback position"""
        # Make this computed depend on subtitle_file_path to trigger updates
        if not self.subtitle_file_path() or not self._subtitle_entries:
            return ""

        current_pos = self.position()

        # Binary search for current subtitle
        left, right = 0, len(self._subtitle_entries) - 1

        while left <= right:
            mid = (left + right) // 2
            entry = self._subtitle_entries[mid]

            if entry.start_time_ms <= current_pos <= entry.end_time_ms:
                return entry.text
            elif current_pos < entry.start_time_ms:
                right = mid - 1
            else:
                left = mid + 1

        return ""

    def _parse_srt_file(self, file_path: str) -> List[SubtitleEntry]:
        """Parse SRT format subtitle file"""
        import re

        entries = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # SRT format: index \n timestamp \n text \n blank line
        # Split by double newlines to get each subtitle block
        blocks = content.strip().split('\n\n')

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            # Line 0: index (we can skip this)
            # Line 1: timestamp
            # Lines 2+: subtitle text

            timestamp_match = re.match(
                r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})',
                lines[1]
            )

            if timestamp_match:
                start_str, end_str = timestamp_match.groups()
                text = '\n'.join(lines[2:])

                entries.append(SubtitleEntry(
                    start_time_ms=self._parse_srt_timestamp(start_str),
                    end_time_ms=self._parse_srt_timestamp(end_str),
                    text=text.strip()
                ))

        return entries

    @staticmethod
    def _parse_srt_timestamp(timestamp: str) -> int:
        """Convert SRT timestamp to milliseconds (00:01:23,456 -> 83456)"""
        h, m, s_ms = timestamp.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
