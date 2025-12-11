"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path
from typing import Generator

# Add parent directory to path so we can import PlayerViewModel
sys.path.insert(0, str(Path(__file__).parent.parent))

from PlayerViewModel import PlayerViewModel


@pytest.fixture
def vm() -> Generator[PlayerViewModel, None, None]:
    """Create a fresh PlayerViewModel instance for each test"""
    yield PlayerViewModel()


@pytest.fixture
def vm_with_video(vm: PlayerViewModel) -> PlayerViewModel:
    """Create a PlayerViewModel with a video loaded"""
    vm.handle_file_loaded("/path/to/test_video.mkv")
    vm.set_duration(120000)  # 2 minutes
    return vm
