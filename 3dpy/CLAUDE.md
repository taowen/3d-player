# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 3D video player built with PyQt6 that supports side-by-side 3D video playback with GPU-accelerated subtitle rendering. The application uses the `reaktiv` library for reactive state management, following an MVVM (Model-View-ViewModel) architecture pattern.

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Running the Application
```bash
# Run the player directly
uv run mkv-player.py

# Or run with Python after activating venv
python mkv-player.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_initialization.py

# Run with verbose output
pytest -v

# Run specific test class or method
pytest tests/test_actions.py::TestPlayerActions::test_toggle_play
```

## Architecture

### MVVM Pattern with Reactive State

The codebase follows a strict MVVM pattern with reactive state management:

- **View** (`MKVPlayer` in mkv-player.py): PyQt6 UI components and event handling. The view is completely declarative and responds to ViewModel state changes via `reaktiv.Effect`.

- **ViewModel** (`PlayerViewModel` in PlayerViewModel.py): Pure business logic with zero Qt dependencies. All state is managed through `reaktiv.Signal` and `reaktiv.Computed` primitives. This makes the ViewModel fully testable without Qt.

- **Model**: Video data is handled by PyQt6's `QMediaPlayer` and `QAudioOutput`, which are wrapped by the View layer.

### Key Architectural Decisions

1. **Reactive State Flow**: The View never directly modifies its own state. Instead:
   - User interactions call ViewModel methods
   - ViewModel updates `Signal` state
   - `Effect` declarations in the View automatically react to state changes
   - This creates a unidirectional data flow that prevents state synchronization bugs

2. **ViewModel Purity**: `PlayerViewModel` has zero PyQt dependencies and uses only standard Python and reaktiv. This enables:
   - Fast, lightweight unit tests without Qt infrastructure
   - Complete separation of business logic from UI framework
   - Easy testing of all state transitions and computed values

3. **GPU-Accelerated Rendering**: Video and subtitles use `QGraphicsView`/`QGraphicsScene` for hardware-accelerated rendering instead of CPU-based widget rendering. This is critical for performance with high-resolution 3D video.

### State Management with reaktiv

The `reaktiv` library provides three primitives:

- `Signal(value)`: Mutable state that can be read with `signal()` and written with `signal.set(value)`
- `Computed(lambda: expr)`: Derived state that automatically recomputes when dependencies change
- `Effect(lambda: expr)`: Side effects that automatically run when dependencies change

Example from the codebase:
```python
# In ViewModel - define reactive state
self.is_playing = Signal(False)
self.play_icon_type = Computed(lambda: "pause" if self.is_playing() else "play")

# In View - react to state changes
Effect(lambda: self.play_button.setIcon(
    self.style().standardIcon(self._get_play_icon())
))
```

## File Structure

- `mkv-player.py`: Main application entry point and View layer (MKVPlayer class)
- `PlayerViewModel.py`: ViewModel with reactive state management and business logic
- `video_widget_graphics.py`: Custom video widget with GPU-accelerated subtitle rendering for side-by-side 3D display
- `tests/`: Comprehensive test suite for PlayerViewModel

## Working with Tests

Tests are organized by concern:
- `test_initialization.py`: Initial state verification
- `test_state_updates.py`: State setter behavior
- `test_computed_values.py`: Computed property calculations
- `test_actions.py`: User action handlers
- `test_utilities.py`: Utility methods like time formatting
- `test_integration.py`: Complex multi-step scenarios

All tests use a `vm` fixture defined in `conftest.py` that provides a fresh PlayerViewModel instance.

## Side-by-Side 3D Subtitle Rendering

The `VideoWidgetGraphics` class displays subtitles in two positions for side-by-side 3D video (left and right halves). Subtitles are rendered as `QGraphicsTextItem` with outline effects and semi-transparent backgrounds, overlaid on the video at z-index 1.

## Type Hints

The codebase uses comprehensive type hints throughout. When modifying code, maintain this standard by:
- Adding return type hints to all functions
- Typing all function parameters
- Using `Optional[T]` for nullable types
- Importing types from `typing` module as needed
