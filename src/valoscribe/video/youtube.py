from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from tqdm import tqdm
from yt_dlp import YoutubeDL

from valoscribe.types.video import DownloadResult
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


def _parse_timestamp(timestamp_str: str) -> Optional[float]:
    """
    Parse YouTube timestamp parameter to seconds.

    Handles formats:
    - "1234" or "1234s" -> 1234 seconds
    - "20m34s" -> 1234 seconds
    - "1h20m34s" -> 4834 seconds

    Returns:
        Seconds as float, or None if parsing failed
    """
    import re

    timestamp_str = timestamp_str.strip()

    # Try simple integer or "1234s" format
    if timestamp_str.endswith('s'):
        try:
            return float(timestamp_str[:-1])
        except ValueError:
            pass
    else:
        try:
            return float(timestamp_str)
        except ValueError:
            pass

    # Try HH:MM:SS or MM:SS format
    if ':' in timestamp_str:
        parts = timestamp_str.split(':')
        try:
            if len(parts) == 2:  # MM:SS
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:  # HH:MM:SS
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            pass

    # Try YouTube format: "1h20m34s"
    pattern = r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?'
    match = re.match(pattern, timestamp_str)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        # Only return if at least one component was matched
        if match.group(1) or match.group(2) or match.group(3):
            return hours * 3600 + minutes * 60 + seconds

    return None


class _TqdmProgressHook:
    """Progress hook for yt-dlp that uses tqdm for clean progress bars."""

    def __init__(self) -> None:
        self.pbar: Optional[tqdm] = None

    def __call__(self, d: Dict[str, Any]) -> None:
        status = d.get("status")

        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)

            if self.pbar is None and total:
                # Initialize progress bar on first call with known total
                self.pbar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading",
                )

            if self.pbar:
                # Update to current position
                self.pbar.n = downloaded
                self.pbar.refresh()

        elif status == "finished":
            if self.pbar:
                self.pbar.close()
                self.pbar = None
            # Single log message after download completes
            log.info("Download complete, merging streams...")


def download_youtube(
    url: str,
    out_dir: str | Path,
    *,
    prefer_height: int = 1080,
    prefer_fps: int = 60,
    prefer_ext: str = "mp4",
    overwrite: bool = False,
    rate_limit: Optional[str] = None,  # e.g., "5M" to limit to ~5 MB/s
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    start_time: Optional[float] = None,  # Start time in seconds for livestream clips
    duration: Optional[float] = None,  # Duration in seconds (default: 45 min)
) -> DownloadResult:
    """
    Download a YouTube VOD to a single merged file suitable for OpenCV.
    Returns metadata including the final output path.

    - Chooses best video ≤ prefer_height and ≤ prefer_fps, prefers MP4 container.
    - Merges audio+video into a single file via ffmpeg.
    - If the exact file exists and overwrite=False, reuses it.

    Timestamped section download (for livestream clips):
    - start_time: Start time in seconds (extracts from URL 't=' param if not provided)
    - duration: Duration to download in seconds (default: 5400s = 90 min)
    - Downloads full video, then trims with ffmpeg (fast stream copy, no re-encode)
    - If duration extends beyond video end, trims to end without error
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse timestamp from URL if not provided (handles t= param in YouTube URLs)
    if start_time is None:
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed.query)
        fragment_params = urllib.parse.parse_qs(parsed.fragment)

        # Check for 't' parameter in query string or fragment
        t_param = query_params.get('t', fragment_params.get('t', [None]))[0]
        if t_param:
            # Handle formats: "1234s", "1234", "20m34s", "1h20m34s"
            start_time = _parse_timestamp(t_param)
            if start_time:
                log.info(f"Extracted start time from URL: {start_time}s")

    # Set default duration if downloading timestamped section
    if start_time is not None and duration is None:
        duration = 5400  # 90 minutes default (handles overtime, downloads to end if shorter)
        log.info(f"Using default duration: {duration}s (90 min)")

    # Filename template; keeping ID avoids title collisions.
    outtmpl = str(out_dir / "%(title).200B-%(id)s.%(ext)s")

    # Format selection logic:
    # Try best video (height/fps/ext filters) + best audio (m4a) -> MP4; fall back to best single.
    fmt = (
        f"bestvideo[height<={prefer_height}][fps<={prefer_fps}][ext={prefer_ext}]"
        "+bestaudio[ext=m4a]/"
        f"best[height<={prefer_height}][fps<={prefer_fps}][ext={prefer_ext}]/"
        "best"
    )

    hooks = [_TqdmProgressHook()]
    if on_progress:
        hooks.append(on_progress)

    ydl_opts = {
        "format": fmt,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "merge_output_format": prefer_ext,   # ensure we end with .mp4 for OpenCV
        "ignoreerrors": False,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": hooks,
        # network/runtime
        "ratelimit": rate_limit,             # None or like "5M"
        # retries (network/transient)
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 5,
        # ytdlp caches format data; safe to keep defaults
    }

    # Add download sections for timestamped clips
    # Note: We'll trim after download using ffmpeg subprocess
    # (yt-dlp's download_ranges doesn't work reliably with all stream types)
    trim_after_download = False
    trim_start = None
    trim_duration = None

    if start_time is not None:
        end_time = start_time + duration
        log.info(f"Will download and trim section: {start_time}s to {end_time}s ({duration}s duration)")
        trim_after_download = True
        trim_start = start_time
        trim_duration = duration

    # If overwrite=False and potential output already exists, short-circuit.
    # We can't know exact final path before download, so we probe info first.
    with YoutubeDL({"quiet": True, "no_warnings": True}) as probe:
        info = probe.extract_info(url, download=False)
    title = info.get("title")
    vid = info.get("id")
    # Guess final path: prefer_ext container after merge
    guessed = out_dir / f"{title}-{vid}.{prefer_ext}"
    if not overwrite and guessed.exists():
        log.info("reusing existing file: %s", guessed)
        return DownloadResult(
            url=url,
            out_path=guessed,
            title=title,
            id=vid,
            height=info.get("height"),
            fps=info.get("fps"),
            duration=info.get("duration"),
        )

    # Perform download
    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        # result may be a dict; after merge, _filename holds the final path
        final_path = Path(ydl.prepare_filename(result))
        # If merge_output_format forced a different extension, adjust
        if final_path.suffix.lower() != f".{prefer_ext}":
            final_path = final_path.with_suffix(f".{prefer_ext}")

    # Double-check the file exists
    if not final_path.exists():
        raise RuntimeError(f"download appeared to succeed but file not found: {final_path}")

    log.info("downloaded: %s", final_path)

    # Trim video if timestamped section was requested
    if trim_after_download:
        import subprocess
        import shutil

        log.info(f"Trimming video: {trim_start}s to {trim_start + trim_duration}s")

        # Create trimmed output path
        trimmed_path = final_path.with_name(f"{final_path.stem}_trimmed{final_path.suffix}")

        # Use ffmpeg to trim (fast, no re-encoding)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-ss", str(trim_start),  # Start time
            "-i", str(final_path),  # Input file
            "-t", str(trim_duration),  # Duration
            "-c", "copy",  # Copy codec (no re-encode, fast!)
            "-avoid_negative_ts", "make_zero",  # Fix timestamp issues
            str(trimmed_path),
        ]

        try:
            # Run ffmpeg
            subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            log.info("Video trimmed successfully")

            # Remove original full file and rename trimmed file
            final_path.unlink()
            shutil.move(str(trimmed_path), str(final_path))
            log.info(f"Replaced with trimmed version: {final_path}")

        except subprocess.CalledProcessError as e:
            log.error(f"ffmpeg trim failed: {e.stderr}")
            # Keep the original file if trim fails
            if trimmed_path.exists():
                trimmed_path.unlink()
            log.warning("Using original untrimmed file")

        except Exception as e:
            log.error(f"Trimming error: {e}")
            if trimmed_path.exists():
                trimmed_path.unlink()
            log.warning("Using original untrimmed file")

    return DownloadResult(
        url=url,
        out_path=final_path,
        title=title or result.get("title"),
        id=vid or result.get("id"),
        height=(result.get("height") or info.get("height")),
        fps=(result.get("fps") or info.get("fps")),
        duration=(result.get("duration") or info.get("duration")),
    )