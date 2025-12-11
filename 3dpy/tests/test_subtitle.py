"""
Tests for subtitle functionality in PlayerViewModel
"""

import pytest
from PlayerViewModel import PlayerViewModel, SubtitleEntry


class TestSubtitleDataStructure:
    """Test SubtitleEntry data structure"""

    def test_subtitle_entry_creation(self):
        entry = SubtitleEntry(
            start_time_ms=1000,
            end_time_ms=3000,
            text="Hello, World!"
        )
        assert entry.start_time_ms == 1000
        assert entry.end_time_ms == 3000
        assert entry.text == "Hello, World!"


class TestSubtitleState:
    """Test subtitle state management"""

    def test_initial_subtitle_state(self, vm):
        assert vm.subtitle_file_path() == ""
        assert not vm.is_subtitle_loaded()
        assert vm.current_subtitle_text() == ""

    def test_subtitle_file_path_signal(self, vm):
        vm.subtitle_file_path.set("/path/to/subtitle.srt")
        assert vm.subtitle_file_path() == "/path/to/subtitle.srt"
        assert vm.is_subtitle_loaded()

    def test_clear_subtitle(self, vm):
        vm.subtitle_file_path.set("/path/to/subtitle.srt")
        vm.clear_subtitle()
        assert vm.subtitle_file_path() == ""
        assert not vm.is_subtitle_loaded()


class TestSubtitleTiming:
    """Test subtitle timing logic"""

    def test_get_subtitle_at_position(self, vm):
        # Manually set subtitle entries
        vm._subtitle_entries = [
            SubtitleEntry(1000, 3000, "First subtitle"),
            SubtitleEntry(5000, 7000, "Second subtitle"),
            SubtitleEntry(10000, 12000, "Third subtitle"),
        ]
        vm.subtitle_file_path.set("dummy.srt")

        # Test before first subtitle
        vm.set_position(500)
        assert vm.current_subtitle_text() == ""

        # Test during first subtitle
        vm.set_position(2000)
        assert vm.current_subtitle_text() == "First subtitle"

        # Test between subtitles
        vm.set_position(4000)
        assert vm.current_subtitle_text() == ""

        # Test during second subtitle
        vm.set_position(6000)
        assert vm.current_subtitle_text() == "Second subtitle"

        # Test during third subtitle
        vm.set_position(11000)
        assert vm.current_subtitle_text() == "Third subtitle"

        # Test after last subtitle
        vm.set_position(15000)
        assert vm.current_subtitle_text() == ""

    def test_subtitle_boundary_conditions(self, vm):
        vm._subtitle_entries = [
            SubtitleEntry(1000, 3000, "Test subtitle"),
        ]
        vm.subtitle_file_path.set("dummy.srt")

        # Test exact start time
        vm.set_position(1000)
        assert vm.current_subtitle_text() == "Test subtitle"

        # Test exact end time
        vm.set_position(3000)
        assert vm.current_subtitle_text() == "Test subtitle"

        # Test just before start
        vm.set_position(999)
        assert vm.current_subtitle_text() == ""

        # Test just after end
        vm.set_position(3001)
        assert vm.current_subtitle_text() == ""

    def test_no_subtitles_loaded(self, vm):
        vm.set_position(5000)
        assert vm.current_subtitle_text() == ""


class TestSubtitleParsing:
    """Test SRT file parsing"""

    def test_parse_srt_timestamp(self):
        vm = PlayerViewModel()

        # Test various timestamps
        assert vm._parse_srt_timestamp("00:00:01,000") == 1000
        assert vm._parse_srt_timestamp("00:01:00,000") == 60000
        assert vm._parse_srt_timestamp("01:00:00,000") == 3600000
        assert vm._parse_srt_timestamp("00:01:23,456") == 83456
        assert vm._parse_srt_timestamp("01:02:03,789") == 3723789

    def test_parse_srt_file(self, vm, tmp_path):
        # Create a temporary SRT file
        srt_content = """1
00:00:01,000 --> 00:00:03,000
First subtitle

2
00:00:05,000 --> 00:00:08,000
Second subtitle
With multiple lines

3
00:00:10,000 --> 00:00:13,000
Third subtitle
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        # Parse the file
        entries = vm._parse_srt_file(str(srt_file))

        # Verify parsed entries
        assert len(entries) == 3

        assert entries[0].start_time_ms == 1000
        assert entries[0].end_time_ms == 3000
        assert entries[0].text == "First subtitle"

        assert entries[1].start_time_ms == 5000
        assert entries[1].end_time_ms == 8000
        assert entries[1].text == "Second subtitle\nWith multiple lines"

        assert entries[2].start_time_ms == 10000
        assert entries[2].end_time_ms == 13000
        assert entries[2].text == "Third subtitle"

    def test_load_subtitle_file_success(self, vm, tmp_path):
        # Create a valid SRT file
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Test subtitle
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        # Load the file
        success = vm.load_subtitle_file(str(srt_file))

        assert success
        assert vm.is_subtitle_loaded()
        assert vm.subtitle_file_path() == str(srt_file)
        assert len(vm._subtitle_entries) == 1

    def test_load_subtitle_file_failure(self, vm):
        # Try to load a non-existent file
        success = vm.load_subtitle_file("/non/existent/file.srt")

        assert not success
        assert not vm.is_subtitle_loaded()
        assert vm.subtitle_file_path() == ""


class TestSubtitleIntegration:
    """Integration tests for subtitle functionality"""

    def test_subtitle_workflow(self, vm, tmp_path):
        # Create a test SRT file
        srt_content = """1
00:00:02,000 --> 00:00:04,000
Hello

2
00:00:06,000 --> 00:00:08,000
World
"""
        srt_file = tmp_path / "workflow.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        # Load subtitle
        assert vm.load_subtitle_file(str(srt_file))

        # Simulate video playback
        vm.set_position(1000)
        assert vm.current_subtitle_text() == ""

        vm.set_position(3000)
        assert vm.current_subtitle_text() == "Hello"

        vm.set_position(5000)
        assert vm.current_subtitle_text() == ""

        vm.set_position(7000)
        assert vm.current_subtitle_text() == "World"

        # Clear subtitle
        vm.clear_subtitle()
        assert vm.current_subtitle_text() == ""
        assert not vm.is_subtitle_loaded()
