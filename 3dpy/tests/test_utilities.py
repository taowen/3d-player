"""
Tests for PlayerViewModel utility methods
"""

import pytest
from PlayerViewModel import PlayerViewModel


class TestTimeFormatting:
    """Test time formatting utility"""

    def test_format_time_zero(self) -> None:
        """Test formatting zero time"""
        assert PlayerViewModel.format_time(0) == "00:00:00"

    def test_format_time_seconds(self) -> None:
        """Test formatting seconds only"""
        assert PlayerViewModel.format_time(1000) == "00:00:01"
        assert PlayerViewModel.format_time(30000) == "00:00:30"
        assert PlayerViewModel.format_time(59000) == "00:00:59"

    def test_format_time_minutes(self) -> None:
        """Test formatting minutes"""
        assert PlayerViewModel.format_time(60000) == "00:01:00"
        assert PlayerViewModel.format_time(90000) == "00:01:30"
        assert PlayerViewModel.format_time(3540000) == "00:59:00"

    def test_format_time_hours(self) -> None:
        """Test formatting hours"""
        assert PlayerViewModel.format_time(3600000) == "01:00:00"
        assert PlayerViewModel.format_time(3661000) == "01:01:01"
        assert PlayerViewModel.format_time(7200000) == "02:00:00"

    def test_format_time_negative(self) -> None:
        """Test formatting negative time returns zero"""
        assert PlayerViewModel.format_time(-1000) == "00:00:00"
        assert PlayerViewModel.format_time(-60000) == "00:00:00"


class TestFileNameExtraction:
    """Test file name extraction utility"""

    def test_get_file_name_empty(self) -> None:
        """Test extracting file name from empty path"""
        assert PlayerViewModel._get_file_name("") == ""

    def test_get_file_name_unix_path(self) -> None:
        """Test extracting file name from Unix path"""
        assert PlayerViewModel._get_file_name("/path/to/video.mkv") == "video.mkv"
        assert PlayerViewModel._get_file_name("/video.mp4") == "video.mp4"

    def test_get_file_name_windows_path(self) -> None:
        """Test extracting file name from Windows path"""
        assert PlayerViewModel._get_file_name("C:\\Users\\test\\video.mp4") == "video.mp4"
        assert PlayerViewModel._get_file_name("D:\\videos\\movie.avi") == "movie.avi"

    def test_get_file_name_no_path(self) -> None:
        """Test extracting file name when there's no path"""
        assert PlayerViewModel._get_file_name("video.mkv") == "video.mkv"


class TestSliderConversions:
    """Test slider position conversions"""

    def test_calculate_position_from_slider_at_start(self, vm: PlayerViewModel) -> None:
        """Test calculating position from slider at start"""
        vm.set_duration(10000)
        assert vm.calculate_position_from_slider(0) == 0

    def test_calculate_position_from_slider_at_middle(self, vm: PlayerViewModel) -> None:
        """Test calculating position from slider at middle"""
        vm.set_duration(10000)
        assert vm.calculate_position_from_slider(500) == 5000

    def test_calculate_position_from_slider_at_end(self, vm: PlayerViewModel) -> None:
        """Test calculating position from slider at end"""
        vm.set_duration(10000)
        assert vm.calculate_position_from_slider(1000) == 10000

    def test_calculate_position_from_slider_no_duration(self, vm: PlayerViewModel) -> None:
        """Test calculating position when duration is zero"""
        vm.set_duration(0)
        assert vm.calculate_position_from_slider(500) == 0

    def test_calculate_position_from_slider_custom_max(self, vm: PlayerViewModel) -> None:
        """Test calculating position with custom slider max"""
        vm.set_duration(10000)
        assert vm.calculate_position_from_slider(50, slider_max=100) == 5000

    def test_get_slider_value_from_position_at_start(self, vm: PlayerViewModel) -> None:
        """Test getting slider value at start"""
        vm.set_duration(10000)
        vm.set_position(0)
        assert vm.get_slider_value_from_position() == 0

    def test_get_slider_value_from_position_at_middle(self, vm: PlayerViewModel) -> None:
        """Test getting slider value at middle"""
        vm.set_duration(10000)
        vm.set_position(5000)
        assert vm.get_slider_value_from_position() == 500

    def test_get_slider_value_from_position_at_end(self, vm: PlayerViewModel) -> None:
        """Test getting slider value at end"""
        vm.set_duration(10000)
        vm.set_position(10000)
        assert vm.get_slider_value_from_position() == 1000


class TestVolumeConversions:
    """Test volume conversion utilities"""

    def test_get_volume_ratio_minimum(self, vm: PlayerViewModel) -> None:
        """Test volume ratio at minimum"""
        vm.set_volume(0)
        assert vm.get_volume_ratio() == 0.0

    def test_get_volume_ratio_middle(self, vm: PlayerViewModel) -> None:
        """Test volume ratio at middle"""
        vm.set_volume(50)
        assert vm.get_volume_ratio() == 0.5

    def test_get_volume_ratio_maximum(self, vm: PlayerViewModel) -> None:
        """Test volume ratio at maximum"""
        vm.set_volume(100)
        assert vm.get_volume_ratio() == 1.0

    def test_get_volume_ratio_custom_value(self, vm: PlayerViewModel) -> None:
        """Test volume ratio at custom value"""
        vm.set_volume(75)
        assert vm.get_volume_ratio() == 0.75


class TestSliderUpdateCondition:
    """Test slider update condition"""

    def test_should_update_slider_when_not_dragging(self, vm: PlayerViewModel) -> None:
        """Test slider should update when not dragging"""
        assert vm.should_update_slider() == True

    def test_should_not_update_slider_when_dragging(self, vm: PlayerViewModel) -> None:
        """Test slider should not update when dragging"""
        vm.start_dragging()
        assert vm.should_update_slider() == False

    def test_should_update_slider_after_drag_ends(self, vm: PlayerViewModel) -> None:
        """Test slider should update after drag ends"""
        vm.start_dragging()
        vm.stop_dragging()
        assert vm.should_update_slider() == True
