"""QGraphicsView based video widget with GPU-accelerated subtitle overlay"""
from typing import Optional
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsTextItem
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPen, QBrush, QPainterPath
from PyQt6.QtMultimedia import QMediaPlayer


class SubtitleTextItem(QGraphicsTextItem):
    """Subtitle text item with outline effect"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setDefaultTextColor(QColor(255, 255, 255))
        self.setFont(QFont("Arial", 24, QFont.Weight.Bold))

        self.background_color = QColor(0, 0, 0, 180)
        self.outline_color = QColor(0, 0, 0)
        self.outline_width = 3

    def paint(self, painter, option, widget=None):
        rect = self.boundingRect()
        padding = 10
        bg_rect = rect.adjusted(-padding, -padding, padding, padding)
        painter.fillRect(bg_rect, self.background_color)

        path = QPainterPath()
        path.addText(rect.topLeft(), self.font(), self.toPlainText())

        painter.setPen(QPen(self.outline_color, self.outline_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setPen(QPen(self.defaultTextColor()))
        painter.drawPath(path)


class VideoWidgetGraphics(QGraphicsView):
    """Video player widget with GPU-accelerated subtitles"""

    def __init__(self, parent: Optional[object] = None, dual_mode: bool = True):
        super().__init__(parent)

        self.dual_mode = dual_mode

        self.setStyleSheet("background-color: black; border: none;")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.video_item = QGraphicsVideoItem()
        self.video_item.setZValue(0)
        self.scene.addItem(self.video_item)

        if dual_mode:
            self.left_subtitle = SubtitleTextItem()
            self.left_subtitle.setZValue(1)
            self.left_subtitle.hide()
            self.scene.addItem(self.left_subtitle)

            self.right_subtitle = SubtitleTextItem()
            self.right_subtitle.setZValue(1)
            self.right_subtitle.hide()
            self.scene.addItem(self.right_subtitle)
        else:
            self.subtitle = SubtitleTextItem()
            self.subtitle.setZValue(1)
            self.subtitle.hide()
            self.scene.addItem(self.subtitle)

    def get_video_sink(self):
        return self.video_item

    def show_subtitle(self, text: str) -> None:
        if not text:
            self.hide_subtitle()
            return

        if self.dual_mode:
            self.left_subtitle.setPlainText(text)
            self.right_subtitle.setPlainText(text)
            self.left_subtitle.show()
            self.right_subtitle.show()
            self._update_subtitle_positions()
        else:
            self.subtitle.setPlainText(text)
            self.subtitle.show()
            self._update_subtitle_positions()

    def hide_subtitle(self) -> None:
        if self.dual_mode:
            self.left_subtitle.hide()
            self.right_subtitle.hide()
        else:
            self.subtitle.hide()

    def _update_subtitle_positions(self) -> None:
        if not self.video_item.size().isValid():
            return

        video_size = self.video_item.size()
        video_width = video_size.width()
        video_height = video_size.height()
        margin_bottom = 40

        if self.dual_mode:
            self._position_dual_subtitles(video_width, video_height, margin_bottom)
        else:
            self._position_single_subtitle(video_width, video_height, margin_bottom)

    def _position_dual_subtitles(self, video_width: float, video_height: float, margin_bottom: int) -> None:
        left_subtitle_rect = self.left_subtitle.boundingRect()
        left_x = (video_width / 4) - (left_subtitle_rect.width() / 2)
        left_y = video_height - margin_bottom - left_subtitle_rect.height()
        self.left_subtitle.setPos(left_x, left_y)

        right_subtitle_rect = self.right_subtitle.boundingRect()
        right_x = (video_width * 3 / 4) - (right_subtitle_rect.width() / 2)
        right_y = video_height - margin_bottom - right_subtitle_rect.height()
        self.right_subtitle.setPos(right_x, right_y)

    def _position_single_subtitle(self, video_width: float, video_height: float, margin_bottom: int) -> None:
        subtitle_rect = self.subtitle.boundingRect()
        x = (video_width - subtitle_rect.width()) / 2
        y = video_height - margin_bottom - subtitle_rect.height()
        self.subtitle.setPos(x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._update_subtitle_positions()
