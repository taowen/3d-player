"""QGraphicsView based video widget with GPU-accelerated subtitle overlay"""
from typing import Optional
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsTextItem
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor


class SubtitleTextItem(QGraphicsTextItem):
    """Subtitle text item for side-by-side 3D video"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setDefaultTextColor(QColor(255, 255, 255))
        # Font size will be dynamically set by VideoWidgetGraphics._update_subtitle_font_size()
        # Use Microsoft YaHei for better Chinese character rendering on Windows
        self.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Normal))


class VideoWidgetGraphics(QGraphicsView):
    """Video player widget with GPU-accelerated subtitles for side-by-side 3D video"""

    def __init__(self, parent: Optional[object] = None):
        super().__init__(parent)

        self.setStyleSheet("background-color: black; border: none;")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.video_item = QGraphicsVideoItem()
        self.video_item.setZValue(0)
        self.scene.addItem(self.video_item)

        self.left_subtitle = SubtitleTextItem()
        self.left_subtitle.setZValue(1)
        self.left_subtitle.setFlag(QGraphicsTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.left_subtitle.hide()
        self.scene.addItem(self.left_subtitle)

        self.right_subtitle = SubtitleTextItem()
        self.right_subtitle.setZValue(1)
        self.right_subtitle.setFlag(QGraphicsTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.right_subtitle.hide()
        self.scene.addItem(self.right_subtitle)

    def get_video_sink(self):
        return self.video_item

    def show_subtitle(self, text: str) -> None:
        if not text:
            self.hide_subtitle()
            return

        self.left_subtitle.setPlainText(text)
        self.right_subtitle.setPlainText(text)
        self.left_subtitle.show()
        self.right_subtitle.show()
        self._update_subtitle_positions()

    def hide_subtitle(self) -> None:
        self.left_subtitle.hide()
        self.right_subtitle.hide()

    def _update_subtitle_positions(self) -> None:
        if not self.video_item.size().isValid():
            return

        video_size = self.video_item.size()
        video_width = video_size.width()
        video_height = video_size.height()
        margin_bottom = 40

        self._position_subtitles(video_width, video_height, margin_bottom)

    def _position_subtitles(self, video_width: float, video_height: float, margin_bottom: int) -> None:
        left_subtitle_rect = self.left_subtitle.boundingRect()
        left_x = (video_width / 4) - (left_subtitle_rect.width() / 2)
        left_y = video_height - margin_bottom - left_subtitle_rect.height()
        self.left_subtitle.setPos(left_x, left_y)

        right_subtitle_rect = self.right_subtitle.boundingRect()
        right_x = (video_width * 3 / 4) - (right_subtitle_rect.width() / 2)
        right_y = video_height - margin_bottom - right_subtitle_rect.height()
        self.right_subtitle.setPos(right_x, right_y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._update_subtitle_positions()
