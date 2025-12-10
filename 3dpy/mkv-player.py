#!/usr/bin/env python3
"""
MKV Video Player
A simple video player built with PyQt6 Multimedia
"""

import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QStyle
)
from PyQt6.QtCore import Qt, QUrl, QTimer, QEvent
from PyQt6.QtGui import QPalette, QColor, QKeyEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget


class MKVPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MKV Player")
        self.setGeometry(100, 100, 1000, 600)

        # Media player and audio output
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # Video widget
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)

        # Connect signals
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)

        # Create UI
        self.create_ui()

        # Track if user is dragging slider
        self.is_dragging = False

        # Fullscreen state
        self.is_fullscreen = False
        self.normal_geometry = None

        # Timer to hide controls in fullscreen
        self.hide_controls_timer = QTimer()
        self.hide_controls_timer.timeout.connect(self.hide_controls)
        self.hide_controls_timer.setSingleShot(True)

        # Enable mouse tracking for fullscreen control visibility
        self.setMouseTracking(True)
        self.video_widget.setMouseTracking(True)
        self.video_widget.installEventFilter(self)

    def create_ui(self):
        """Create the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        central_widget.setLayout(layout)

        # Video widget
        self.video_widget.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_widget, stretch=1)  # Give video widget all available space

        # Control panel
        control_layout = QHBoxLayout()

        # Open file button
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_file)
        control_layout.addWidget(self.open_button)

        # Play button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_button)

        # Stop button
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop)
        control_layout.addWidget(self.stop_button)

        # Fullscreen button
        self.fullscreen_button = QPushButton()
        self.fullscreen_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton))
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        control_layout.addWidget(self.fullscreen_button)

        # Volume label
        volume_label = QLabel("Volume:")
        control_layout.addWidget(volume_label)

        # Volume slider
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.set_volume)
        control_layout.addWidget(self.volume_slider)

        # Store control panel for fullscreen hide/show
        self.control_panel = QWidget()
        self.control_panel.setLayout(control_layout)
        layout.addWidget(self.control_panel, stretch=0)  # No stretch for controls

        # Progress bar layout
        progress_layout = QHBoxLayout()

        # Time label (current time)
        self.time_label = QLabel("00:00:00")
        progress_layout.addWidget(self.time_label)

        # Position slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setMaximum(1000)
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        self.position_slider.sliderMoved.connect(self.set_position)
        progress_layout.addWidget(self.position_slider)

        # Duration label
        self.duration_label = QLabel("00:00:00")
        progress_layout.addWidget(self.duration_label)

        # Store progress panel for fullscreen hide/show
        self.progress_panel = QWidget()
        self.progress_panel.setLayout(progress_layout)
        layout.addWidget(self.progress_panel, stretch=0)  # No stretch for progress bar

        # File name label
        self.file_label = QLabel("No file loaded")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.file_label)

    def open_file(self):
        """Open a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            str(Path.home()),
            "Video Files (*.mkv *.mp4 *.avi *.mov *.wmv *.flv *.webm);;All Files (*.*)"
        )

        if file_path:
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.play()

            # Update file label
            file_name = os.path.basename(file_path)
            self.file_label.setText(f"Playing: {file_name}")

    def toggle_play(self):
        """Toggle play/pause"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause()
        else:
            self.play()

    def play(self):
        """Play the video"""
        self.media_player.play()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def pause(self):
        """Pause the video"""
        self.media_player.pause()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def stop(self):
        """Stop the video"""
        self.media_player.stop()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.position_slider.setValue(0)
        self.time_label.setText("00:00:00")

    def set_volume(self, value):
        """Set the volume"""
        self.audio_output.setVolume(value / 100.0)

    def on_slider_pressed(self):
        """Handle slider press"""
        self.is_dragging = True

    def on_slider_released(self):
        """Handle slider release"""
        self.is_dragging = False
        self.set_position(self.position_slider.value())

    def set_position(self, position):
        """Set the playback position"""
        if self.is_dragging:
            # Convert slider position to media position
            duration = self.media_player.duration()
            if duration > 0:
                new_position = int((position / 1000.0) * duration)
                self.media_player.setPosition(new_position)

    def position_changed(self, position):
        """Handle position changed signal"""
        if not self.is_dragging:
            duration = self.media_player.duration()
            if duration > 0:
                self.position_slider.setValue(int((position / duration) * 1000))
            self.time_label.setText(self.format_time(position))

    def duration_changed(self, duration):
        """Handle duration changed signal"""
        self.duration_label.setText(self.format_time(duration))

    def format_time(self, milliseconds):
        """Format time in milliseconds to HH:MM:SS"""
        if milliseconds < 0:
            return "00:00:00"

        seconds = milliseconds // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self):
        """Enter fullscreen mode"""
        self.is_fullscreen = True
        self.normal_geometry = self.geometry()

        # Hide menu bar, status bar, and file label
        self.file_label.hide()

        # Show fullscreen
        self.showFullScreen()

        # Update button icon
        self.fullscreen_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton))

        # Start timer to hide controls
        self.hide_controls_timer.start(3000)  # Hide after 3 seconds

    def exit_fullscreen(self):
        """Exit fullscreen mode"""
        self.is_fullscreen = False

        # Stop hide timer
        self.hide_controls_timer.stop()

        # Show all controls
        self.control_panel.show()
        self.progress_panel.show()
        self.file_label.show()
        self.setCursor(Qt.CursorShape.ArrowCursor)

        # Restore normal window
        self.showNormal()
        if self.normal_geometry:
            self.setGeometry(self.normal_geometry)

        # Update button icon
        self.fullscreen_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton))

    def hide_controls(self):
        """Hide controls in fullscreen mode"""
        if self.is_fullscreen:
            self.control_panel.hide()
            self.progress_panel.hide()
            self.setCursor(Qt.CursorShape.BlankCursor)

    def show_controls(self):
        """Show controls in fullscreen mode"""
        if self.is_fullscreen:
            self.control_panel.show()
            self.progress_panel.show()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            # Restart timer to hide controls
            self.hide_controls_timer.start(3000)

    def eventFilter(self, obj, event):
        """Event filter for video widget to handle double-click"""
        if obj == self.video_widget:
            if event.type() == QEvent.Type.MouseButtonDblClick:
                self.toggle_fullscreen()
                return True
            elif event.type() == QEvent.Type.MouseMove and self.is_fullscreen:
                self.show_controls()
                return False
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Escape and self.is_fullscreen:
            self.exit_fullscreen()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_play()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event"""
        self.media_player.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)

    player = MKVPlayer()
    player.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
