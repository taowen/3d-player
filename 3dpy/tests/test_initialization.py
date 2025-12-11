"""
Tests for PlayerViewModel initialization
"""

from PlayerViewModel import PlayerViewModel


class TestPlayerViewModelInitialization:
    """Test ViewModel initialization"""

    def test_initial_playback_state(self, vm: PlayerViewModel) -> None:
        """Test that playback state initializes correctly"""
        assert vm.position() == 0
        assert vm.duration() == 0
        assert vm.is_playing() == False
        assert vm.is_dragging() == False

    def test_initial_ui_state(self, vm: PlayerViewModel) -> None:
        """Test that UI state initializes correctly"""
        assert vm.volume() == 50
        assert vm.is_fullscreen() == False
        assert vm.file_path() == ""

    def test_initial_computed_values(self, vm: PlayerViewModel) -> None:
        """Test that computed values work correctly on initialization"""
        assert vm.progress_percent() == 0
        assert vm.current_time_text() == "00:00:00"
        assert vm.duration_time_text() == "00:00:00"
        assert vm.file_name() == ""
        assert vm.is_file_loaded() == False
        assert vm.file_display_text() == "No file loaded"

    def test_initial_icon_types(self, vm: PlayerViewModel) -> None:
        """Test that icon types are correct on initialization"""
        assert vm.play_icon_type() == "play"
        assert vm.fullscreen_icon_type() == "maximize"
