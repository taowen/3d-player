"""
MKV Video Player
A simple video player built with PyQt6 Multimedia with reaktiv state management

Run: uv run mkv-player.py
"""

import sys
from pathlib import Path
from typing import Optional, Any
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QStyle
)
from PyQt6.QtCore import Qt, QUrl, QTimer, QEvent, QRect
from PyQt6.QtGui import QPalette, QColor, QKeyEvent, QCloseEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from reaktiv import Effect

from PlayerViewModel import PlayerViewModel
from video_widget_with_subtitle import VideoWidgetWithSubtitle


class MKVPlayer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MKV Player (Reactive)")
        self.setGeometry(100, 100, 1000, 600)

        # === ViewModel (Business Logic) ===
        self.vm: PlayerViewModel = PlayerViewModel()

        # Media player and audio output
        self.media_player: QMediaPlayer = QMediaPlayer()
        self.audio_output: QAudioOutput = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # Video widget with subtitle support
        self.video_widget: VideoWidgetWithSubtitle = VideoWidgetWithSubtitle(parent=None, dual_mode=True)
        self.media_player.setVideoSink(self.video_widget.get_video_sink())

        # Connect Qt signals to ViewModel reactive state
        self.media_player.positionChanged.connect(lambda pos: self.vm.set_position(pos))
        self.media_player.durationChanged.connect(lambda dur: self.vm.set_duration(dur))
        self.media_player.playbackStateChanged.connect(
            lambda state: self.vm.set_playing_state(state == QMediaPlayer.PlaybackState.PlayingState)
        )

        # Fullscreen state
        self.normal_geometry: Optional[QRect] = None

        # Timer to hide controls in fullscreen (initialize BEFORE setup_effects)
        self.hide_controls_timer: QTimer = QTimer()
        self.hide_controls_timer.timeout.connect(self.hide_controls)
        self.hide_controls_timer.setSingleShot(True)

        # Create UI
        self.create_ui()

        # Setup reactive effects for UI updates (must be AFTER all initialization)
        self.setup_effects()

        # Enable mouse tracking for fullscreen control visibility
        self.setMouseTracking(True)
        self.video_widget.setMouseTracking(True)
        self.video_widget.installEventFilter(self)

        # UI widgets (type declarations)
        self.open_button: QPushButton
        self.play_button: QPushButton
        self.stop_button: QPushButton
        self.fullscreen_button: QPushButton
        self.volume_slider: QSlider
        self.position_slider: QSlider
        self.time_label: QLabel
        self.duration_label: QLabel
        self.file_label: QLabel
        self.control_panel: QWidget
        self.progress_panel: QWidget

    def create_ui(self) -> None:
        """Create the user interface"""
        # Central widget
        central_widget: QWidget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        central_widget.setLayout(layout)

        # Video widget
        layout.addWidget(self.video_widget, stretch=1)  # Give video widget all available space

        # Show test subtitle
        self.video_widget.show_subtitle("Hello World")

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

    def setup_effects(self) -> None:
        """Setup reactive effects for UI updates"""
        # Update progress slider when position changes (but not while dragging)
        Effect(lambda: (
            self.position_slider.setValue(self.vm.get_slider_value_from_position())
            if self.vm.should_update_slider() else None
        ))

        # Update time label
        Effect(lambda: self.time_label.setText(self.vm.current_time_text()))

        # Update duration label
        Effect(lambda: self.duration_label.setText(self.vm.duration_time_text()))

        # Update file name label
        Effect(lambda: self.file_label.setText(self.vm.file_display_text()))

        # Update play button icon
        Effect(lambda: self.play_button.setIcon(
            self.style().standardIcon(self._get_play_icon())
        ))

        # Update fullscreen button icon
        Effect(lambda: self.fullscreen_button.setIcon(
            self.style().standardIcon(self._get_fullscreen_icon())
        ))

        # Handle fullscreen UI changes
        Effect(lambda: self.update_fullscreen_ui() if self.vm.is_fullscreen() else self.update_normal_ui())

    def _get_play_icon(self) -> QStyle.StandardPixmap:
        """Get the appropriate play/pause icon from ViewModel state"""
        icon_type = self.vm.play_icon_type()
        return (QStyle.StandardPixmap.SP_MediaPause if icon_type == "pause"
                else QStyle.StandardPixmap.SP_MediaPlay)

    def _get_fullscreen_icon(self) -> QStyle.StandardPixmap:
        """Get the appropriate fullscreen icon from ViewModel state"""
        icon_type = self.vm.fullscreen_icon_type()
        return (QStyle.StandardPixmap.SP_TitleBarNormalButton if icon_type == "restore"
                else QStyle.StandardPixmap.SP_TitleBarMaxButton)

    def open_file(self) -> None:
        """Open a video file"""
        file_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            str(Path.home()),
            "Video Files (*.mkv *.mp4 *.avi *.mov *.wmv *.flv *.webm);;All Files (*.*)"
        )

        if file_path_str:
            self.media_player.setSource(QUrl.fromLocalFile(file_path_str))
            self.vm.handle_file_loaded(file_path_str)
            self.play()

    def toggle_play(self) -> None:
        """Toggle play/pause"""
        if self.vm.toggle_play():
            self.play()
        else:
            self.pause()

    def play(self) -> None:
        """Play the video"""
        self.media_player.play()
        # is_playing state is automatically updated via playbackStateChanged signal

    def pause(self) -> None:
        """Pause the video"""
        self.media_player.pause()
        # is_playing state is automatically updated via playbackStateChanged signal

    def stop(self) -> None:
        """Stop the video"""
        self.media_player.stop()
        self.vm.handle_stop_action()

    def set_volume(self, value: int) -> None:
        """Set the volume"""
        self.vm.set_volume(value)
        self.audio_output.setVolume(self.vm.get_volume_ratio())

    def on_slider_pressed(self) -> None:
        """Handle slider press"""
        self.vm.start_dragging()

    def on_slider_released(self) -> None:
        """Handle slider release"""
        self.vm.stop_dragging()
        self.set_position(self.position_slider.value())

    def set_position(self, slider_value: int) -> None:
        """Set the playback position from slider value"""
        new_position = self.vm.calculate_position_from_slider(slider_value)
        self.media_player.setPosition(new_position)

    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode"""
        self.vm.toggle_fullscreen()

    def update_fullscreen_ui(self) -> None:
        """Update UI for fullscreen mode (called by reactive effect)"""
        if not hasattr(self, 'control_panel'):
            return  # UI not ready yet

        self.normal_geometry = self.geometry()
        self.file_label.hide()
        self.showFullScreen()
        self.hide_controls_timer.start(3000)  # Hide after 3 seconds

    def update_normal_ui(self) -> None:
        """Update UI for normal mode (called by reactive effect)"""
        if not hasattr(self, 'control_panel'):
            return  # UI not ready yet

        self.hide_controls_timer.stop()
        self.control_panel.show()
        self.progress_panel.show()
        self.file_label.show()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.showNormal()
        if self.normal_geometry:
            self.setGeometry(self.normal_geometry)

    def hide_controls(self) -> None:
        """Hide controls in fullscreen mode"""
        if self.vm.is_fullscreen():
            self.control_panel.hide()
            self.progress_panel.hide()
            self.setCursor(Qt.CursorShape.BlankCursor)

    def show_controls(self) -> None:
        """Show controls in fullscreen mode"""
        if self.vm.is_fullscreen():
            self.control_panel.show()
            self.progress_panel.show()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            # Restart timer to hide controls
            self.hide_controls_timer.start(3000)

    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        """Event filter for video widget to handle double-click"""
        if obj == self.video_widget:
            if event.type() == QEvent.Type.MouseButtonDblClick:
                self.toggle_fullscreen()
                return True
            elif event.type() == QEvent.Type.MouseMove and self.vm.is_fullscreen():
                self.show_controls()
                return False
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events"""
        if event.key() == Qt.Key.Key_F11:
            self.vm.handle_f11_key()
        elif event.key() == Qt.Key.Key_Escape:
            self.vm.handle_escape_key()
        elif event.key() == Qt.Key.Key_Space:
            if self.vm.handle_space_key():
                self.play()
            else:
                self.pause()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event"""
        self.media_player.stop()
        event.accept()


def main() -> None:
    app: QApplication = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")
    dark_palette: QPalette = QPalette()
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

    player: MKVPlayer = MKVPlayer()
    player.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
