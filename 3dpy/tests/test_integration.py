"""
Integration tests for PlayerViewModel
Tests realistic usage scenarios combining multiple features
"""

from PlayerViewModel import PlayerViewModel


class TestCompletePlaybackScenario:
    """Test a complete video playback scenario"""

    def test_full_playback_workflow(self, vm: PlayerViewModel) -> None:
        """Test complete workflow from loading to stopping"""
        # Load file
        vm.handle_file_loaded("/path/to/video.mkv")
        assert vm.file_path() == "/path/to/video.mkv"
        assert vm.is_playing() == True
        assert vm.file_display_text() == "Playing: video.mkv"

        # Set duration
        vm.set_duration(120000)  # 2 minutes
        assert vm.duration_time_text() == "00:02:00"

        # Update position
        vm.set_position(60000)  # 1 minute
        assert vm.current_time_text() == "00:01:00"
        assert vm.progress_percent() == 500  # 50%

        # Pause
        vm.toggle_play()
        assert vm.is_playing() == False
        assert vm.play_icon_type() == "play"

        # Resume
        vm.toggle_play()
        assert vm.is_playing() == True
        assert vm.play_icon_type() == "pause"

        # Stop
        vm.handle_stop_action()
        assert vm.is_playing() == False


class TestSliderInteractionScenario:
    """Test slider dragging and seeking scenario"""

    def test_slider_drag_workflow(self, vm: PlayerViewModel) -> None:
        """Test complete slider drag and seek workflow"""
        vm.set_duration(10000)
        vm.set_position(3000)

        # User starts dragging
        vm.start_dragging()
        assert vm.should_update_slider() == False

        # Position changes during drag shouldn't trigger slider update
        vm.set_position(4000)
        assert vm.is_dragging() == True
        assert vm.should_update_slider() == False

        # User releases slider at 70% position
        vm.stop_dragging()
        assert vm.should_update_slider() == True

        # Calculate and apply new position from slider
        new_pos = vm.calculate_position_from_slider(700)  # 70%
        vm.set_position(new_pos)
        assert vm.position() == 7000

        # Verify slider value matches position
        slider_val = vm.get_slider_value_from_position()
        assert slider_val == 700

    def test_seeking_during_playback(self, vm_with_video: PlayerViewModel) -> None:
        """Test seeking while video is playing"""
        # Start playing
        vm_with_video.set_playing_state(True)
        assert vm_with_video.is_playing() == True

        # User drags slider
        vm_with_video.start_dragging()
        vm_with_video.set_position(30000)  # Seek to 30 seconds

        # Release slider
        vm_with_video.stop_dragging()

        # Should still be playing after seek
        assert vm_with_video.is_playing() == True
        assert vm_with_video.position() == 30000


class TestFullscreenWorkflow:
    """Test fullscreen interaction workflow"""

    def test_enter_exit_fullscreen_with_f11(self, vm: PlayerViewModel) -> None:
        """Test entering and exiting fullscreen with F11"""
        # Enter fullscreen with F11
        vm.handle_f11_key()
        assert vm.is_fullscreen() == True
        assert vm.fullscreen_icon_type() == "restore"

        # Exit with F11 again
        vm.handle_f11_key()
        assert vm.is_fullscreen() == False
        assert vm.fullscreen_icon_type() == "maximize"

    def test_exit_fullscreen_with_escape(self, vm: PlayerViewModel) -> None:
        """Test exiting fullscreen with Escape key"""
        # Enter fullscreen
        vm.toggle_fullscreen()
        assert vm.is_fullscreen() == True

        # Exit with Escape
        vm.handle_escape_key()
        assert vm.is_fullscreen() == False

    def test_fullscreen_with_toggle(self, vm: PlayerViewModel) -> None:
        """Test fullscreen with toggle method"""
        # Enter with toggle
        vm.toggle_fullscreen()
        assert vm.is_fullscreen() == True
        assert vm.fullscreen_icon_type() == "restore"

        # Exit with toggle
        vm.toggle_fullscreen()
        assert vm.is_fullscreen() == False
        assert vm.fullscreen_icon_type() == "maximize"

    def test_escape_key_in_normal_mode(self, vm: PlayerViewModel) -> None:
        """Test that Escape key in normal mode doesn't affect state"""
        assert vm.is_fullscreen() == False
        vm.handle_escape_key()
        assert vm.is_fullscreen() == False


class TestKeyboardControlsWorkflow:
    """Test keyboard control workflow"""

    def test_space_key_playback_control(self, vm: PlayerViewModel) -> None:
        """Test space key for play/pause control"""
        # Space to play
        result = vm.handle_space_key()
        assert result == True
        assert vm.is_playing() == True
        assert vm.play_icon_type() == "pause"

        # Space to pause
        result = vm.handle_space_key()
        assert result == False
        assert vm.is_playing() == False
        assert vm.play_icon_type() == "play"

    def test_combined_keyboard_shortcuts(self, vm: PlayerViewModel) -> None:
        """Test multiple keyboard shortcuts in sequence"""
        # Start playback with space
        vm.handle_space_key()
        assert vm.is_playing() == True

        # Enter fullscreen with F11
        vm.handle_f11_key()
        assert vm.is_fullscreen() == True

        # Pause with space
        vm.handle_space_key()
        assert vm.is_playing() == False
        assert vm.is_fullscreen() == True  # Still in fullscreen

        # Exit fullscreen with Escape
        vm.handle_escape_key()
        assert vm.is_fullscreen() == False
        assert vm.is_playing() == False  # Still paused


class TestVolumeControl:
    """Test volume control scenarios"""

    def test_volume_adjustment_during_playback(self, vm_with_video: PlayerViewModel) -> None:
        """Test adjusting volume while playing"""
        vm_with_video.set_playing_state(True)

        # Adjust volume
        vm_with_video.set_volume(75)
        assert vm_with_video.get_volume_ratio() == 0.75
        assert vm_with_video.is_playing() == True

        # Mute
        vm_with_video.set_volume(0)
        assert vm_with_video.get_volume_ratio() == 0.0
        assert vm_with_video.is_playing() == True

        # Max volume
        vm_with_video.set_volume(100)
        assert vm_with_video.get_volume_ratio() == 1.0


class TestFileLoadingSequence:
    """Test file loading and switching scenarios"""

    def test_load_multiple_files(self, vm: PlayerViewModel) -> None:
        """Test loading multiple files in sequence"""
        # Load first file
        vm.handle_file_loaded("/path/to/video1.mkv")
        assert vm.file_name() == "video1.mkv"
        assert vm.is_playing() == True

        # Simulate playback
        vm.set_duration(60000)
        vm.set_position(30000)

        # Load second file (should reset and auto-play)
        vm.handle_file_loaded("/path/to/video2.mp4")
        assert vm.file_name() == "video2.mp4"
        assert vm.is_playing() == True
        assert vm.file_display_text() == "Playing: video2.mp4"

    def test_stop_and_load_new_file(self, vm_with_video: PlayerViewModel) -> None:
        """Test stopping current file and loading new one"""
        # Stop current playback
        vm_with_video.handle_stop_action()
        assert vm_with_video.is_playing() == False

        # Load new file
        vm_with_video.handle_file_loaded("/path/to/new_video.mkv")
        assert vm_with_video.file_name() == "new_video.mkv"
        assert vm_with_video.is_playing() == True


class TestProgressTracking:
    """Test progress tracking during playback"""

    def test_progress_updates(self, vm: PlayerViewModel) -> None:
        """Test progress calculation as position changes"""
        vm.set_duration(100000)  # 100 seconds

        # 0%
        vm.set_position(0)
        assert vm.progress_percent() == 0
        assert vm.get_slider_value_from_position() == 0

        # 25%
        vm.set_position(25000)
        assert vm.progress_percent() == 250
        assert vm.get_slider_value_from_position() == 250

        # 50%
        vm.set_position(50000)
        assert vm.progress_percent() == 500
        assert vm.get_slider_value_from_position() == 500

        # 75%
        vm.set_position(75000)
        assert vm.progress_percent() == 750
        assert vm.get_slider_value_from_position() == 750

        # 100%
        vm.set_position(100000)
        assert vm.progress_percent() == 1000
        assert vm.get_slider_value_from_position() == 1000


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_position_beyond_duration(self, vm: PlayerViewModel) -> None:
        """Test handling position beyond duration"""
        vm.set_duration(10000)
        vm.set_position(15000)  # Beyond duration

        # Should still calculate progress (even if > 100%)
        assert vm.position() == 15000
        assert vm.progress_percent() == 1500

    def test_zero_duration_operations(self, vm: PlayerViewModel) -> None:
        """Test operations when duration is zero"""
        vm.set_duration(0)
        vm.set_position(5000)

        assert vm.progress_percent() == 0
        assert vm.calculate_position_from_slider(500) == 0

    def test_rapid_state_changes(self, vm: PlayerViewModel) -> None:
        """Test rapid state changes"""
        # Rapid play/pause
        for _ in range(10):
            vm.toggle_play()

        # Should end up paused (started from not playing)
        assert vm.is_playing() == False

        # Rapid fullscreen toggle
        for _ in range(10):
            vm.toggle_fullscreen()

        # Should end up in normal mode
        assert vm.is_fullscreen() == False
