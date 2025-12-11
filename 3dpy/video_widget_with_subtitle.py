"""
Custom video widget with subtitle rendering on video frames
通过处理视频帧直接绘制字幕
"""
from typing import Optional
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QImage, QFont, QPen, QColor, QPaintEvent
from PyQt6.QtMultimedia import QVideoFrame, QVideoSink


class VideoWidgetWithSubtitle(QWidget):
    """支持字幕渲染的视频显示组件"""

    def __init__(self, parent: Optional[QWidget] = None, dual_mode: bool = True) -> None:
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")

        self.dual_mode = dual_mode
        self.current_frame: Optional[QImage] = None
        self.subtitle_text: str = ""

        # Video sink to receive video frames
        self.video_sink = QVideoSink()
        self.video_sink.videoFrameChanged.connect(self.on_video_frame)

    def on_video_frame(self, frame: QVideoFrame) -> None:
        """接收视频帧并转换为 QImage"""
        if frame.isValid():
            # Map the frame to access pixel data
            frame.map(QVideoFrame.MapMode.ReadOnly)

            # Convert to QImage
            image = frame.toImage()

            frame.unmap()

            if not image.isNull():
                self.current_frame = image
                self.update()  # Trigger repaint

    def paintEvent(self, event: QPaintEvent) -> None:
        """绘制视频帧和字幕"""
        painter = QPainter(self)

        if self.current_frame:
            # Scale video frame to widget size
            scaled_image = self.current_frame.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Center the image
            x = (self.width() - scaled_image.width()) // 2
            y = (self.height() - scaled_image.height()) // 2
            painter.drawImage(x, y, scaled_image)

            # Draw subtitles
            if self.subtitle_text:
                self.draw_subtitles(painter, scaled_image.width(), scaled_image.height(), x, y)
        else:
            # No video frame, just draw black background
            painter.fillRect(self.rect(), Qt.GlobalColor.black)

    def draw_subtitles(self, painter: QPainter, video_width: int, video_height: int, offset_x: int, offset_y: int) -> None:
        """在视频上绘制字幕"""
        # Setup font
        font = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(font)

        # Calculate text metrics
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self.subtitle_text)
        text_height = metrics.height()

        padding = 10
        margin_bottom = 40

        if self.dual_mode:
            # Draw left subtitle (1/4 position from left)
            left_center_x = offset_x + video_width // 4
            self.draw_single_subtitle(
                painter,
                left_center_x,
                offset_y + video_height - margin_bottom,
                text_width,
                text_height,
                padding
            )

            # Draw right subtitle (3/4 position from left)
            right_center_x = offset_x + (video_width * 3) // 4
            self.draw_single_subtitle(
                painter,
                right_center_x,
                offset_y + video_height - margin_bottom,
                text_width,
                text_height,
                padding
            )
        else:
            # Draw single subtitle (center)
            center_x = offset_x + video_width // 2
            self.draw_single_subtitle(
                painter,
                center_x,
                offset_y + video_height - margin_bottom,
                text_width,
                text_height,
                padding
            )

    def draw_single_subtitle(self, painter: QPainter, center_x: int, bottom_y: int,
                            text_width: int, text_height: int, padding: int) -> None:
        """绘制单个字幕"""
        # Draw background rectangle
        bg_rect = QRect(
            center_x - text_width // 2 - padding,
            bottom_y - text_height - padding,
            text_width + padding * 2,
            text_height + padding * 2
        )

        painter.fillRect(bg_rect, QColor(0, 0, 0, 180))

        # Draw text outline (black)
        painter.setPen(QPen(QColor(0, 0, 0), 3))
        text_rect = QRect(
            center_x - text_width // 2,
            bottom_y - text_height,
            text_width,
            text_height
        )
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.subtitle_text)

        # Draw text (white)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.subtitle_text)

    def show_subtitle(self, text: str) -> None:
        """显示字幕"""
        self.subtitle_text = text
        self.update()

    def hide_subtitle(self) -> None:
        """隐藏字幕"""
        self.subtitle_text = ""
        self.update()

    def get_video_sink(self) -> QVideoSink:
        """获取 video sink 用于连接 QMediaPlayer"""
        return self.video_sink
