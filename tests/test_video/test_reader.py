"""Unit tests for video reader module."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import pytest
import numpy as np

from valoscribe.video.reader import FileVideoSource, VideoReader, read_video_frames
from valoscribe.types.video import FrameInfo


class TestFileVideoSource:
    """Tests for FileVideoSource class."""

    @pytest.fixture
    def mock_video_capture(self):
        """Mock cv2.VideoCapture."""
        with patch("valoscribe.video.reader.cv2.VideoCapture") as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
            mock_cap.get.side_effect = lambda prop: {
                3: 100,  # CAP_PROP_POS_FRAMES
                5: 60.0,  # CAP_PROP_FPS
                7: 1000,  # CAP_PROP_FRAME_COUNT
                3: 1920,  # CAP_PROP_FRAME_WIDTH
                4: 1080,  # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0)
            mock_cap.set.return_value = True
            mock_cap_class.return_value = mock_cap
            yield mock_cap_class, mock_cap

    def test_init_with_valid_file(self, tmp_path, mock_video_capture):
        """Test initialization with a valid video file."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        source = FileVideoSource(video_file)

        assert source.video_path == video_file
        assert source._cap is not None
        mock_video_capture[0].assert_called_once_with(str(video_file))

    def test_init_with_missing_file(self, tmp_path):
        """Test initialization with non-existent file raises FileNotFoundError."""
        video_file = tmp_path / "missing.mp4"

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            FileVideoSource(video_file)

    def test_init_with_unopenable_file(self, tmp_path, mock_video_capture):
        """Test initialization when cv2 can't open file raises RuntimeError."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        # Mock isOpened to return False
        mock_video_capture[1].isOpened.return_value = False

        with pytest.raises(RuntimeError, match="Could not open video"):
            FileVideoSource(video_file)

    def test_read_frame_success(self, tmp_path, mock_video_capture):
        """Test reading a frame successfully."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        source = FileVideoSource(video_file)
        success, frame = source.read_frame()

        assert success is True
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        mock_video_capture[1].read.assert_called_once()

    def test_read_frame_failure(self, tmp_path, mock_video_capture):
        """Test reading frame when read fails."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_video_capture[1].read.return_value = (False, None)

        source = FileVideoSource(video_file)
        success, frame = source.read_frame()

        assert success is False
        assert frame is None

    def test_get_position(self, tmp_path, mock_video_capture):
        """Test getting current frame position."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        source = FileVideoSource(video_file)

        # After initialization, update mock to return specific position
        mock_video_capture[1].get.side_effect = None
        mock_video_capture[1].get.return_value = 42

        position = source.get_position()

        assert position == 42
        # CAP_PROP_POS_FRAMES = 1
        assert mock_video_capture[1].get.call_args_list[-1][0][0] == 1

    def test_set_position(self, tmp_path, mock_video_capture):
        """Test seeking to specific frame."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        source = FileVideoSource(video_file)
        success = source.set_position(100)

        assert success is True
        # CAP_PROP_POS_FRAMES = 1
        mock_video_capture[1].set.assert_called_with(1, 100)

    def test_get_fps(self, tmp_path, mock_video_capture):
        """Test getting video FPS."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_video_capture[1].get.return_value = 60.0

        source = FileVideoSource(video_file)
        fps = source.get_fps()

        assert fps == 60.0
        # CAP_PROP_FPS = 5
        mock_video_capture[1].get.assert_called_with(5)

    def test_get_frame_count(self, tmp_path, mock_video_capture):
        """Test getting total frame count."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_video_capture[1].get.return_value = 1000

        source = FileVideoSource(video_file)
        count = source.get_frame_count()

        assert count == 1000
        # CAP_PROP_FRAME_COUNT = 7
        mock_video_capture[1].get.assert_called_with(7)

    def test_get_width(self, tmp_path, mock_video_capture):
        """Test getting frame width."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_video_capture[1].get.return_value = 1920

        source = FileVideoSource(video_file)
        width = source.get_width()

        assert width == 1920
        # CAP_PROP_FRAME_WIDTH = 3
        mock_video_capture[1].get.assert_called_with(3)

    def test_get_height(self, tmp_path, mock_video_capture):
        """Test getting frame height."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_video_capture[1].get.return_value = 1080

        source = FileVideoSource(video_file)
        height = source.get_height()

        assert height == 1080
        # CAP_PROP_FRAME_HEIGHT = 4
        mock_video_capture[1].get.assert_called_with(4)

    def test_release(self, tmp_path, mock_video_capture):
        """Test releasing video source."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        source = FileVideoSource(video_file)
        source.release()

        mock_video_capture[1].release.assert_called_once()
        assert source._cap is None


class TestVideoReader:
    """Tests for VideoReader class."""

    @pytest.fixture
    def mock_source(self):
        """Create a mock VideoSource."""
        mock = Mock()
        mock.get_fps.return_value = 60.0
        mock.get_frame_count.return_value = 600  # 10 seconds at 60fps
        mock.get_width.return_value = 1920
        mock.get_height.return_value = 1080
        mock.get_position.return_value = 0
        mock.set_position.return_value = True

        # Create frames that can be iterated
        frames = []
        for i in range(600):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frames.append((True, frame))
        frames.append((False, None))  # End of video

        mock.read_frame.side_effect = frames
        return mock

    def test_initialization(self, tmp_path):
        """Test VideoReader initialization."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("valoscribe.video.reader.FileVideoSource") as mock_source_class:
            mock_source = Mock()
            mock_source.get_fps.return_value = 60.0
            mock_source.get_frame_count.return_value = 600
            mock_source.get_width.return_value = 1920
            mock_source.get_height.return_value = 1080
            mock_source.get_position.return_value = 0
            mock_source_class.return_value = mock_source

            reader = VideoReader(video_file)

            assert reader.fps == 60.0
            assert reader.frame_count == 600
            assert reader.width == 1920
            assert reader.height == 1080
            assert reader.duration_sec == 10.0

    def test_iteration_basic(self, mock_source):
        """Test basic frame iteration."""
        reader = VideoReader(mock_source)

        frame_count = 0
        for frame_info in reader:
            assert isinstance(frame_info, FrameInfo)
            assert isinstance(frame_info.frame, np.ndarray)
            assert frame_info.frame_number == frame_count
            frame_count += 1
            if frame_count >= 10:  # Only test first 10 frames
                break

        assert frame_count == 10

    def test_fps_filter(self, mock_source):
        """Test FPS filtering (e.g., process 5 fps from 60 fps video)."""
        reader = VideoReader(mock_source, fps_filter=5.0)

        assert reader._frame_interval == 12  # 60 / 5 = 12

        frame_count = 0
        for frame_info in reader:
            # Frames should be at intervals of 12
            assert frame_info.frame_number % 12 == 0
            frame_count += 1
            if frame_count >= 5:
                break

        assert frame_count == 5

    def test_start_time_filter(self, mock_source):
        """Test starting from specific timestamp."""
        reader = VideoReader(mock_source, start_time_sec=2.0)

        # Should seek to frame 120 (2 seconds * 60 fps)
        mock_source.set_position.assert_called_with(120)

    def test_end_time_filter(self, mock_source):
        """Test ending at specific timestamp."""
        reader = VideoReader(mock_source, end_time_sec=0.1)  # 0.1 seconds = 6 frames

        frame_count = 0
        for _ in reader:
            frame_count += 1
            if frame_count >= 100:  # Safety limit
                break

        # Should stop around frame 6 (0.1s * 60fps)
        assert frame_count <= 6

    def test_frame_filter(self, mock_source):
        """Test custom frame filter."""
        # Filter that only accepts every other frame
        frame_filter = Mock(side_effect=lambda frame: mock_source.get_position() % 2 == 0)

        reader = VideoReader(mock_source, frame_filter=frame_filter)

        frame_count = 0
        for _ in reader:
            frame_count += 1
            if frame_count >= 5:
                break

        # Frame filter should have been called
        assert frame_filter.call_count > 0

    def test_progress_callback(self, mock_source):
        """Test progress callback is called for each frame."""
        callback = Mock()
        reader = VideoReader(mock_source, progress_callback=callback)

        frame_count = 0
        for _ in reader:
            frame_count += 1
            if frame_count >= 5:
                break

        # Callback should be called for each processed frame
        assert callback.call_count == 5
        # Check that FrameInfo was passed to callback
        assert isinstance(callback.call_args_list[0][0][0], FrameInfo)

    def test_context_manager(self, tmp_path):
        """Test VideoReader as context manager."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("valoscribe.video.reader.FileVideoSource") as mock_source_class:
            mock_source = Mock()
            mock_source.get_fps.return_value = 60.0
            mock_source.get_frame_count.return_value = 600
            mock_source.get_width.return_value = 1920
            mock_source.get_height.return_value = 1080
            mock_source.get_position.return_value = 0
            mock_source_class.return_value = mock_source

            with VideoReader(video_file) as reader:
                assert reader is not None

            # Release should be called when exiting context
            mock_source.release.assert_called_once()

    def test_seek(self, mock_source):
        """Test seeking to specific timestamp."""
        reader = VideoReader(mock_source)

        success = reader.seek(5.0)  # Seek to 5 seconds

        assert success is True
        # Should seek to frame 300 (5s * 60fps)
        mock_source.set_position.assert_called_with(300)

    def test_get_frame_at(self, mock_source):
        """Test getting frame at specific timestamp without advancing iterator."""
        # Setup mock to return a frame
        test_frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        mock_source.read_frame.return_value = (True, test_frame)
        mock_source.get_position.return_value = 120

        reader = VideoReader(mock_source)

        frame_info = reader.get_frame_at(2.0)  # Get frame at 2 seconds

        assert frame_info is not None
        assert isinstance(frame_info, FrameInfo)
        assert frame_info.timestamp_sec == 2.0
        assert frame_info.timestamp_ms == 2000.0
        # Position should be restored after get_frame_at
        assert mock_source.set_position.call_count >= 2  # Set and restore

    def test_get_frame_at_failure(self, mock_source):
        """Test get_frame_at when read fails."""
        reader = VideoReader(mock_source)

        # Clear side_effect and set failure return value
        mock_source.read_frame.side_effect = None
        mock_source.read_frame.return_value = (False, None)

        frame_info = reader.get_frame_at(2.0)

        assert frame_info is None

    def test_close(self, mock_source):
        """Test closing video reader."""
        reader = VideoReader(mock_source)
        reader.close()

        mock_source.release.assert_called_once()

    def test_repr(self, mock_source):
        """Test string representation."""
        reader = VideoReader(mock_source)
        repr_str = repr(reader)

        assert "VideoReader" in repr_str
        assert "60.00" in repr_str  # FPS
        assert "1920x1080" in repr_str  # Resolution
        assert "10.00s" in repr_str  # Duration


class TestReadVideoFrames:
    """Tests for read_video_frames convenience function."""

    def test_read_video_frames(self, tmp_path):
        """Test convenience function for reading frames."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("valoscribe.video.reader.VideoReader") as mock_reader_class:
            # Use MagicMock to support __iter__
            mock_reader = MagicMock()
            # Create a simple iterator
            test_frames = [
                FrameInfo(
                    frame_number=i,
                    timestamp_ms=i * 16.67,
                    timestamp_sec=i / 60.0,
                    frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
                )
                for i in range(5)
            ]
            mock_reader.__iter__.return_value = iter(test_frames)
            mock_reader.__enter__.return_value = mock_reader
            mock_reader.__exit__.return_value = None
            mock_reader_class.return_value = mock_reader

            frames = list(read_video_frames(video_file, fps_filter=5.0))

            assert len(frames) == 5
            assert all(isinstance(f, FrameInfo) for f in frames)

            # Check that VideoReader was initialized with correct params
            mock_reader_class.assert_called_once()
            call_kwargs = mock_reader_class.call_args[1]
            assert call_kwargs["fps_filter"] == 5.0

    def test_read_video_frames_with_time_range(self, tmp_path):
        """Test convenience function with time range."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("valoscribe.video.reader.VideoReader") as mock_reader_class:
            # Use MagicMock to support __iter__
            mock_reader = MagicMock()
            mock_reader.__iter__.return_value = iter([])
            mock_reader.__enter__.return_value = mock_reader
            mock_reader.__exit__.return_value = None
            mock_reader_class.return_value = mock_reader

            list(read_video_frames(
                video_file,
                start_time_sec=1.0,
                end_time_sec=5.0,
            ))

            # Check parameters
            call_kwargs = mock_reader_class.call_args[1]
            assert call_kwargs["start_time_sec"] == 1.0
            assert call_kwargs["end_time_sec"] == 5.0
