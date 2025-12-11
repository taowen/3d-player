"""
Tests for PlayerViewModel computed values
"""

from PlayerViewModel import PlayerViewModel


class TestProgressCalculation:
    """Test progress percentage calculation"""

    def test_progress_with_no_duration(self, vm: PlayerViewModel) -> None:
        """Test progress when duration is zero"""
        vm.set_position(1000)
        vm.set_duration(0)
        assert vm.progress_percent() == 0

    def test_progress_at_50_percent(self, vm: PlayerViewModel) -> None:
        """Test progress at 50%"""
        vm.set_duration(10000)
        vm.set_position(5000)
        assert vm.progress_percent() == 500

    def test_progress_at_100_percent(self, vm: PlayerViewModel) -> None:
        """Test progress at 100%"""
        vm.set_duration(10000)
        vm.set_position(10000)
        assert vm.progress_percent() == 1000

    def test_progress_at_0_percent(self, vm: PlayerViewModel) -> None:
        """Test progress at 0%"""
        vm.set_duration(10000)
        vm.set_position(0)
        assert vm.progress_percent() == 0


class TestTimeFormatting:
    """Test time formatting for display"""

    def test_current_time_text_zero(self, vm: PlayerViewModel) -> None:
        """Test current time at zero"""
        vm.set_position(0)
        assert vm.current_time_text() == "00:00:00"

    def test_current_time_text_seconds(self, vm: PlayerViewModel) -> None:
        """Test current time in seconds"""
        vm.set_position(5000)  # 5 seconds
        assert vm.current_time_text() == "00:00:05"

    def test_current_time_text_minutes(self, vm: PlayerViewModel) -> None:
        """Test current time in minutes"""
        vm.set_position(65000)  # 1 minute 5 seconds
        assert vm.current_time_text() == "00:01:05"

    def test_current_time_text_hours(self, vm: PlayerViewModel) -> None:
        """Test current time in hours"""
        vm.set_position(3661000)  # 1 hour 1 minute 1 second
        assert vm.current_time_text() == "01:01:01"

    def test_duration_time_text(self, vm: PlayerViewModel) -> None:
        """Test duration time formatting"""
        vm.set_duration(120000)  # 2 minutes
        assert vm.duration_time_text() == "00:02:00"

        vm.set_duration(3600000)  # 1 hour
        assert vm.duration_time_text() == "01:00:00"


class TestFileNameExtraction:
    """Test file name extraction from paths"""

    def test_file_name_unix_path(self, vm: PlayerViewModel) -> None:
        """Test file name extraction from Unix path"""
        vm.set_file_path("/path/to/video.mkv")
        assert vm.file_name() == "video.mkv"

    def test_file_name_windows_path(self, vm: PlayerViewModel) -> None:
        """Test file name extraction from Windows path"""
        vm.set_file_path("C:\\Users\\test\\video.mp4")
        assert vm.file_name() == "video.mp4"

    def test_file_name_empty_path(self, vm: PlayerViewModel) -> None:
        """Test file name extraction from empty path"""
        vm.set_file_path("")
        assert vm.file_name() == ""

    def test_file_name_only(self, vm: PlayerViewModel) -> None:
        """Test file name without path"""
        vm.set_file_path("video.mkv")
        assert vm.file_name() == "video.mkv"


class TestFileLoadedStatus:
    """Test file loaded status"""

    def test_is_file_loaded_initially_false(self, vm: PlayerViewModel) -> None:
        """Test that file is not loaded initially"""
        assert vm.is_file_loaded() == False

    def test_is_file_loaded_after_setting_path(self, vm: PlayerViewModel) -> None:
        """Test file loaded status after setting path"""
        vm.set_file_path("/path/to/video.mkv")
        assert vm.is_file_loaded() == True

    def test_is_file_loaded_after_clearing_path(self, vm: PlayerViewModel) -> None:
        """Test file loaded status after clearing path"""
        vm.set_file_path("/path/to/video.mkv")
        vm.set_file_path("")
        assert vm.is_file_loaded() == False


class TestDisplayText:
    """Test display text generation"""

    def test_file_display_text_no_file(self, vm: PlayerViewModel) -> None:
        """Test display text when no file is loaded"""
        assert vm.file_display_text() == "No file loaded"

    def test_file_display_text_with_file(self, vm: PlayerViewModel) -> None:
        """Test display text when file is loaded"""
        vm.set_file_path("/path/to/video.mkv")
        assert vm.file_display_text() == "Playing: video.mkv"


class TestIconTypes:
    """Test icon type determination"""

    def test_play_icon_type_when_not_playing(self, vm: PlayerViewModel) -> None:
        """Test play icon type when not playing"""
        assert vm.play_icon_type() == "play"

    def test_play_icon_type_when_playing(self, vm: PlayerViewModel) -> None:
        """Test play icon type when playing"""
        vm.set_playing_state(True)
        assert vm.play_icon_type() == "pause"

    def test_fullscreen_icon_type_normal_mode(self, vm: PlayerViewModel) -> None:
        """Test fullscreen icon type in normal mode"""
        assert vm.fullscreen_icon_type() == "maximize"

    def test_fullscreen_icon_type_fullscreen_mode(self, vm: PlayerViewModel) -> None:
        """Test fullscreen icon type in fullscreen mode"""
        vm.set_fullscreen(True)
        assert vm.fullscreen_icon_type() == "restore"
