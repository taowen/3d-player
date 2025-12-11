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
from video_widget_graphics import VideoWidgetGraphics


class MKVPlayer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MKV Player (Reactive)")
        self.setGeometry(100, 100, 1000, 600)

        self.vm: PlayerViewModel = PlayerViewModel()

        self.media_player: QMediaPlayer = QMediaPlayer()
        self.audio_output: QAudioOutput = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        self.video_widget: VideoWidgetGraphics = VideoWidgetGraphics(parent=None)
        self.media_player.setVideoOutput(self.video_widget.get_video_sink())

        self._connect_media_player_signals()

        self.normal_geometry: Optional[QRect] = None
        self.hide_controls_timer: QTimer = self._create_fullscreen_timer()

        self.create_ui()
        self.setup_effects()
        self._enable_fullscreen_mouse_tracking()

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

    def _connect_media_player_signals(self) -> None:
        self.media_player.positionChanged.connect(lambda pos: self.vm.set_position(pos))
        self.media_player.durationChanged.connect(lambda dur: self.vm.set_duration(dur))
        self.media_player.playbackStateChanged.connect(
            lambda state: self.vm.set_playing_state(state == QMediaPlayer.PlaybackState.PlayingState)
        )

    def _create_fullscreen_timer(self) -> QTimer:
        timer = QTimer()
        timer.timeout.connect(self.hide_controls)
        timer.setSingleShot(True)
        return timer

    def _enable_fullscreen_mouse_tracking(self) -> None:
        self.setMouseTracking(True)
        self.video_widget.setMouseTracking(True)
        self.video_widget.installEventFilter(self)

    def create_ui(self) -> None:
        central_widget: QWidget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        central_widget.setLayout(layout)

        layout.addWidget(self.video_widget, stretch=1)
        self.video_widget.show_subtitle("Hello")

        self.control_panel = self._create_control_panel()
        layout.addWidget(self.control_panel, stretch=0)

        self.progress_panel = self._create_progress_panel()
        layout.addWidget(self.progress_panel, stretch=0)

        self.file_label = QLabel("No file loaded")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.file_label)

    def _create_control_panel(self) -> QWidget:
        control_layout = QHBoxLayout()

        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_file)
        control_layout.addWidget(self.open_button)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_button)

        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop)
        control_layout.addWidget(self.stop_button)

        self.fullscreen_button = QPushButton()
        self.fullscreen_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton))
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        control_layout.addWidget(self.fullscreen_button)

        volume_label = QLabel("Volume:")
        control_layout.addWidget(volume_label)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.set_volume)
        control_layout.addWidget(self.volume_slider)

        panel = QWidget()
        panel.setLayout(control_layout)
        return panel

    def _create_progress_panel(self) -> QWidget:
        progress_layout = QHBoxLayout()

        self.time_label = QLabel("00:00:00")
        progress_layout.addWidget(self.time_label)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setMaximum(1000)
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        self.position_slider.sliderMoved.connect(self.set_position)
        progress_layout.addWidget(self.position_slider)

        self.duration_label = QLabel("00:00:00")
        progress_layout.addWidget(self.duration_label)

        panel = QWidget()
        panel.setLayout(progress_layout)
        return panel

    def setup_effects(self) -> None:
        Effect(lambda: (
            self.position_slider.setValue(self.vm.get_slider_value_from_position())
            if self.vm.should_update_slider() else None
        ))
        Effect(lambda: self.time_label.setText(self.vm.current_time_text()))
        Effect(lambda: self.duration_label.setText(self.vm.duration_time_text()))
        Effect(lambda: self.file_label.setText(self.vm.file_display_text()))
        Effect(lambda: self.play_button.setIcon(
            self.style().standardIcon(self._get_play_icon())
        ))
        Effect(lambda: self.fullscreen_button.setIcon(
            self.style().standardIcon(self._get_fullscreen_icon())
        ))
        Effect(lambda: self.update_fullscreen_ui() if self.vm.is_fullscreen() else self.update_normal_ui())

    def _get_play_icon(self) -> QStyle.StandardPixmap:
        icon_type = self.vm.play_icon_type()
        return (QStyle.StandardPixmap.SP_MediaPause if icon_type == "pause"
                else QStyle.StandardPixmap.SP_MediaPlay)

    def _get_fullscreen_icon(self) -> QStyle.StandardPixmap:
        icon_type = self.vm.fullscreen_icon_type()
        return (QStyle.StandardPixmap.SP_TitleBarNormalButton if icon_type == "restore"
                else QStyle.StandardPixmap.SP_TitleBarMaxButton)

    def open_file(self) -> None:
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
        if self.vm.toggle_play():
            self.play()
        else:
            self.pause()

    def play(self) -> None:
        self.media_player.play()

    def pause(self) -> None:
        self.media_player.pause()

    def stop(self) -> None:
        self.media_player.stop()
        self.vm.handle_stop_action()

    def set_volume(self, value: int) -> None:
        self.vm.set_volume(value)
        self.audio_output.setVolume(self.vm.get_volume_ratio())

    def on_slider_pressed(self) -> None:
        self.vm.start_dragging()

    def on_slider_released(self) -> None:
        self.vm.stop_dragging()
        self.set_position(self.position_slider.value())

    def set_position(self, slider_value: int) -> None:
        new_position = self.vm.calculate_position_from_slider(slider_value)
        self.media_player.setPosition(new_position)

    def toggle_fullscreen(self) -> None:
        self.vm.toggle_fullscreen()

    def update_fullscreen_ui(self) -> None:
        if not hasattr(self, 'control_panel'):
            return

        self.normal_geometry = self.geometry()
        self.file_label.hide()
        self.showFullScreen()
        self.hide_controls_timer.start(3000)

    def update_normal_ui(self) -> None:
        if not hasattr(self, 'control_panel'):
            return

        self.hide_controls_timer.stop()
        self.control_panel.show()
        self.progress_panel.show()
        self.file_label.show()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.showNormal()
        if self.normal_geometry:
            self.setGeometry(self.normal_geometry)

    def hide_controls(self) -> None:
        if self.vm.is_fullscreen():
            self.control_panel.hide()
            self.progress_panel.hide()
            self.setCursor(Qt.CursorShape.BlankCursor)

    def show_controls(self) -> None:
        if self.vm.is_fullscreen():
            self.control_panel.show()
            self.progress_panel.show()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.hide_controls_timer.start(3000)

    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        if obj == self.video_widget:
            if event.type() == QEvent.Type.MouseButtonDblClick:
                self.toggle_fullscreen()
                return True
            elif event.type() == QEvent.Type.MouseMove and self.vm.is_fullscreen():
                self.show_controls()
                return False
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
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
        self.media_player.stop()
        event.accept()


def _create_dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    return palette


def main() -> None:
    app: QApplication = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_create_dark_palette())

    player: MKVPlayer = MKVPlayer()
    player.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
