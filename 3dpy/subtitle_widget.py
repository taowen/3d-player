"""
Subtitle overlay widget for side-by-side 3D video player
支持左右分屏显示镜像字幕
"""
from typing import Optional
from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor


class SubtitleOverlay(QWidget):
    """透明的字幕叠加层 - 支持左右双字幕"""

    def __init__(self, parent: Optional[QWidget] = None, dual_mode: bool = True) -> None:
        """
        Args:
            parent: 父窗口
            dual_mode: 是否启用双字幕模式（左右镜像）
        """
        super().__init__(parent)

        # 设置为透明背景，鼠标事件穿透
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self.dual_mode = dual_mode

        # 字幕样式
        subtitle_style = """
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 150);
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 18pt;
                font-weight: bold;
            }
        """

        if dual_mode:
            # 左右双字幕布局
            main_layout = QHBoxLayout()

            # 左侧字幕
            left_container = QVBoxLayout()
            self.left_subtitle = QLabel()
            self.left_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.left_subtitle.setWordWrap(True)
            self.left_subtitle.setStyleSheet(subtitle_style)
            left_container.addStretch()
            left_container.addWidget(self.left_subtitle)
            left_container.setContentsMargins(20, 20, 20, 20)

            # 右侧字幕
            right_container = QVBoxLayout()
            self.right_subtitle = QLabel()
            self.right_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.right_subtitle.setWordWrap(True)
            self.right_subtitle.setStyleSheet(subtitle_style)
            right_container.addStretch()
            right_container.addWidget(self.right_subtitle)
            right_container.setContentsMargins(20, 20, 20, 20)

            # 添加到主布局
            main_layout.addLayout(left_container, stretch=1)
            main_layout.addLayout(right_container, stretch=1)
            main_layout.setSpacing(0)
            main_layout.setContentsMargins(0, 0, 0, 0)

            self.left_subtitle.hide()
            self.right_subtitle.hide()
        else:
            # 单字幕模式（居中底部）
            main_layout = QVBoxLayout()
            self.subtitle_label = QLabel()
            self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.subtitle_label.setWordWrap(True)
            self.subtitle_label.setStyleSheet(subtitle_style)

            main_layout.addStretch()
            main_layout.addWidget(self.subtitle_label)
            main_layout.setContentsMargins(50, 50, 50, 50)

            self.subtitle_label.hide()

        self.setLayout(main_layout)

    def show_subtitle(self, text: str) -> None:
        """显示字幕文本"""
        if text:
            if self.dual_mode:
                self.left_subtitle.setText(text)
                self.right_subtitle.setText(text)
                self.left_subtitle.show()
                self.right_subtitle.show()
            else:
                self.subtitle_label.setText(text)
                self.subtitle_label.show()
        else:
            self.hide_subtitle()

    def hide_subtitle(self) -> None:
        """隐藏字幕"""
        if self.dual_mode:
            self.left_subtitle.hide()
            self.right_subtitle.hide()
        else:
            self.subtitle_label.hide()

    def set_font_size(self, size: int) -> None:
        """设置字幕字体大小"""
        if self.dual_mode:
            font_left = self.left_subtitle.font()
            font_left.setPointSize(size)
            self.left_subtitle.setFont(font_left)

            font_right = self.right_subtitle.font()
            font_right.setPointSize(size)
            self.right_subtitle.setFont(font_right)
        else:
            font = self.subtitle_label.font()
            font.setPointSize(size)
            self.subtitle_label.setFont(font)

    def set_dual_mode(self, enabled: bool) -> None:
        """动态切换单/双字幕模式"""
        # 注意：此方法需要重建布局，建议在初始化时就确定模式
        pass
