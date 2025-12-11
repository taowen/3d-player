"""
Tests for PlayerViewModel action methods
"""

from PlayerViewModel import PlayerViewModel


class TestPlaybackActions:
    """Test playback control actions"""

    def test_toggle_play_from_stopped(self, vm: PlayerViewModel) -> None:
        """Test toggle play from stopped state"""
        result = vm.toggle_play()
        assert result == True
        assert vm.is_playing() == True

    def test_toggle_play_from_playing(self, vm: PlayerViewModel) -> None:
        """Test toggle play from playing state"""
        vm.set_playing_state(True)
        result = vm.toggle_play()
        assert result == False
        assert vm.is_playing() == False

    def test_handle_stop_action(self, vm: PlayerViewModel) -> None:
        """Test stop action resets playing state"""
        vm.set_playing_state(True)
        vm.handle_stop_action()
        assert vm.is_playing() == False


class TestFullscreenActions:
    """Test fullscreen control actions"""

    def test_toggle_fullscreen_to_on(self, vm: PlayerViewModel) -> None:
        """Test toggling fullscreen on"""
        vm.toggle_fullscreen()
        assert vm.is_fullscreen() == True

    def test_toggle_fullscreen_to_off(self, vm: PlayerViewModel) -> None:
        """Test toggling fullscreen off"""
        vm.set_fullscreen(True)
        vm.toggle_fullscreen()
        assert vm.is_fullscreen() == False


class TestDraggingActions:
    """Test slider dragging actions"""

    def test_start_dragging(self, vm: PlayerViewModel) -> None:
        """Test starting drag operation"""
        vm.start_dragging()
        assert vm.is_dragging() == True

    def test_stop_dragging(self, vm: PlayerViewModel) -> None:
        """Test stopping drag operation"""
        vm.start_dragging()
        vm.stop_dragging()
        assert vm.is_dragging() == False


class TestFileLoadingActions:
    """Test file loading actions"""

    def test_handle_file_loaded(self, vm: PlayerViewModel) -> None:
        """Test file loaded handler sets path and auto-plays"""
        vm.handle_file_loaded("/path/to/video.mkv")
        assert vm.file_path() == "/path/to/video.mkv"
        assert vm.is_playing() == True

    def test_handle_file_loaded_with_different_path(self, vm: PlayerViewModel) -> None:
        """Test loading a different file"""
        vm.handle_file_loaded("/path/to/video1.mkv")
        vm.handle_file_loaded("/path/to/video2.mkv")
        assert vm.file_path() == "/path/to/video2.mkv"
        assert vm.is_playing() == True


class TestKeyboardActions:
    """Test keyboard shortcut actions"""

    def test_handle_escape_key_when_not_fullscreen(self, vm: PlayerViewModel) -> None:
        """Test escape key when not in fullscreen does nothing"""
        vm.handle_escape_key()
        assert vm.is_fullscreen() == False

    def test_handle_escape_key_when_fullscreen(self, vm: PlayerViewModel) -> None:
        """Test escape key exits fullscreen"""
        vm.set_fullscreen(True)
        vm.handle_escape_key()
        assert vm.is_fullscreen() == False

    def test_handle_f11_key_toggle_on(self, vm: PlayerViewModel) -> None:
        """Test F11 key toggles fullscreen on"""
        vm.handle_f11_key()
        assert vm.is_fullscreen() == True

    def test_handle_f11_key_toggle_off(self, vm: PlayerViewModel) -> None:
        """Test F11 key toggles fullscreen off"""
        vm.set_fullscreen(True)
        vm.handle_f11_key()
        assert vm.is_fullscreen() == False

    def test_handle_space_key_to_play(self, vm: PlayerViewModel) -> None:
        """Test space key starts playback"""
        result = vm.handle_space_key()
        assert result == True
        assert vm.is_playing() == True

    def test_handle_space_key_to_pause(self, vm: PlayerViewModel) -> None:
        """Test space key pauses playback"""
        vm.set_playing_state(True)
        result = vm.handle_space_key()
        assert result == False
        assert vm.is_playing() == False
