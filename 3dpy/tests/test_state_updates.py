"""
Tests for PlayerViewModel state updates
"""

from PlayerViewModel import PlayerViewModel


class TestStateUpdates:
    """Test state update methods"""

    def test_set_position(self, vm: PlayerViewModel) -> None:
        """Test position updates"""
        vm.set_position(5000)
        assert vm.position() == 5000

        vm.set_position(10000)
        assert vm.position() == 10000

    def test_set_duration(self, vm: PlayerViewModel) -> None:
        """Test duration updates"""
        vm.set_duration(120000)
        assert vm.duration() == 120000

        vm.set_duration(60000)
        assert vm.duration() == 60000

    def test_set_playing_state(self, vm: PlayerViewModel) -> None:
        """Test playing state updates"""
        vm.set_playing_state(True)
        assert vm.is_playing() == True

        vm.set_playing_state(False)
        assert vm.is_playing() == False

    def test_set_volume_valid_values(self, vm: PlayerViewModel) -> None:
        """Test volume updates with valid values"""
        vm.set_volume(0)
        assert vm.volume() == 0

        vm.set_volume(100)
        assert vm.volume() == 100

        vm.set_volume(75)
        assert vm.volume() == 75

    def test_set_volume_invalid_values(self, vm: PlayerViewModel) -> None:
        """Test volume updates with invalid values"""
        vm.set_volume(50)
        initial_volume = vm.volume()

        # Invalid volumes should not update
        vm.set_volume(-1)
        assert vm.volume() == initial_volume

        vm.set_volume(101)
        assert vm.volume() == initial_volume

    def test_set_dragging(self, vm: PlayerViewModel) -> None:
        """Test dragging state updates"""
        vm.set_dragging(True)
        assert vm.is_dragging() == True

        vm.set_dragging(False)
        assert vm.is_dragging() == False

    def test_set_fullscreen(self, vm: PlayerViewModel) -> None:
        """Test fullscreen state updates"""
        vm.set_fullscreen(True)
        assert vm.is_fullscreen() == True

        vm.set_fullscreen(False)
        assert vm.is_fullscreen() == False

    def test_set_file_path(self, vm: PlayerViewModel) -> None:
        """Test file path updates"""
        vm.set_file_path("/path/to/video.mkv")
        assert vm.file_path() == "/path/to/video.mkv"

        vm.set_file_path("C:\\Users\\test\\video.mp4")
        assert vm.file_path() == "C:\\Users\\test\\video.mp4"
