"""
SRT 字幕文件解析器
"""
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SubtitleEntry:
    """字幕条目"""
    index: int
    start_time: int  # 毫秒
    end_time: int    # 毫秒
    text: str


class SubtitleParser:
    """SRT 字幕解析器"""

    def __init__(self) -> None:
        self.subtitles: List[SubtitleEntry] = []

    def parse_srt(self, file_path: str) -> bool:
        """
        解析 SRT 字幕文件

        Args:
            file_path: SRT 文件路径

        Returns:
            是否解析成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()

            # 按空行分割字幕块
            blocks = re.split(r'\n\s*\n', content.strip())
            self.subtitles.clear()

            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue

                # 解析索引
                try:
                    index = int(lines[0].strip())
                except ValueError:
                    continue

                # 解析时间轴: 00:00:20,000 --> 00:00:24,400
                time_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    lines[1]
                )
                if not time_match:
                    continue

                start_h, start_m, start_s, start_ms = map(int, time_match.groups()[:4])
                end_h, end_m, end_s, end_ms = map(int, time_match.groups()[4:])

                start_time = (start_h * 3600 + start_m * 60 + start_s) * 1000 + start_ms
                end_time = (end_h * 3600 + end_m * 60 + end_s) * 1000 + end_ms

                # 解析字幕文本（可能多行）
                text = '\n'.join(lines[2:])

                self.subtitles.append(SubtitleEntry(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                ))

            return len(self.subtitles) > 0

        except Exception as e:
            print(f"字幕解析错误: {e}")
            return False

    def get_subtitle_at_time(self, position_ms: int) -> Optional[str]:
        """
        获取指定时间点的字幕文本

        Args:
            position_ms: 播放位置（毫秒）

        Returns:
            字幕文本，如果没有则返回 None
        """
        for subtitle in self.subtitles:
            if subtitle.start_time <= position_ms <= subtitle.end_time:
                return subtitle.text
        return None

    def clear(self) -> None:
        """清空字幕数据"""
        self.subtitles.clear()
