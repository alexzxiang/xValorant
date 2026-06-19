"""Utility commands for video processing and analysis."""

import typer
import json
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.video.youtube import download_youtube
from valoscribe.video.reader import VideoReader
from valoscribe.detectors.cropper import Cropper
from valoscribe.utils.logger import setup_logging

app = typer.Typer(help="Utility commands for video processing")
@app.command()
def download(
    url: str = typer.Argument(..., help="YouTube URL of the Valorant VOD (supports timestamped URLs)"),
    output_dir: Path = typer.Option(
        Path("./videos"),
        "--output",
        "-o",
        help="Output directory for downloaded video",
    ),
    height: int = typer.Option(
        1080,
        "--height",
        "-h",
        help="Preferred video height (default: 1080p)",
    ),
    fps: int = typer.Option(
        60,
        "--fps",
        help="Preferred video FPS (default: 60)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing file if present",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        "-s",
        help="Start time in seconds (for livestream clips, auto-extracted from URL 't=' param)",
    ),
    duration: Optional[float] = typer.Option(
        None,
        "--duration",
        "-d",
        help="Duration in seconds (default: 5400s = 90 min if start_time is set)",
    ),
) -> None:
    """
    Download a Valorant VOD from YouTube.

    Supports full VODs and timestamped livestream clips:
    - Full VOD: valoscribe download "https://youtube.com/watch?v=VIDEO_ID"
    - Timestamped clip: valoscribe download "https://youtube.com/watch?v=VIDEO_ID&t=1234s"
    - Manual time range: valoscribe download URL --start 1234 --duration 1800
    """
    setup_logging()

    try:
        typer.echo(f"Downloading from: {url}")
        typer.echo(f"Target: {height}p @ {fps}fps -> {output_dir}")
        if start_time is not None or 't=' in url:
            typer.echo(f"Timestamped section download enabled")
            if start_time:
                typer.echo(f"  Start: {start_time}s")
            if duration:
                typer.echo(f"  Duration: {duration}s")
            else:
                typer.echo(f"  Duration: 5400s (90 min default)")

        result = download_youtube(
            url=url,
            out_dir=output_dir,
            prefer_height=height,
            prefer_fps=fps,
            overwrite=overwrite,
            start_time=start_time,
            duration=duration,
        )

        typer.secho(f"\nSuccess!", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"File: {result.out_path}")
        typer.echo(f"Title: {result.title}")
        typer.echo(f"Resolution: {result.height}p @ {result.fps}fps")
        if result.duration:
            mins = int(result.duration // 60)
            secs = int(result.duration % 60)
            typer.echo(f"Duration: {mins}m {secs}s")

    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def read(
    video_path: Path = typer.Argument(..., help="Path to the video file to read"),
    save_frames: Optional[Path] = typer.Option(
        None,
        "--save-frames",
        "-s",
        help="Directory to save frames (if not specified, frames are only displayed)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        None,
        "--fps",
        "-f",
        help="Process frames at this FPS (e.g., 5 for 5 frames/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process (for debugging)",
    ),
    frame_format: str = typer.Option(
        "jpg",
        "--format",
        help="Frame image format when saving (jpg, png, etc.)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging to see frame seeking details",
    ),
) -> None:
    """Read and display frames from a Valorant VOD."""
    import logging
    setup_logging(level=logging.DEBUG if debug else logging.INFO)

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")

            if fps:
                typer.echo(f"Processing at: {fps} fps")
            if start_time:
                typer.echo(f"Start time: {start_time}s")
            if end_time:
                typer.echo(f"End time: {end_time}s")
            if max_frames:
                typer.echo(f"Max frames: {max_frames}")

            if save_frames:
                save_frames.mkdir(parents=True, exist_ok=True)
                typer.echo(f"Saving frames to: {save_frames}")

            typer.echo("\nPress 'q' to quit, SPACE to pause/resume\n")

            frame_count = 0
            paused = False

            for frame_info in reader:
                # Check max_frames limit
                if max_frames and frame_count >= max_frames:
                    typer.echo(f"\nReached max frames limit: {max_frames}")
                    break

                # Save frame if requested
                if save_frames:
                    frame_filename = f"frame_{frame_info.frame_number:06d}.{frame_format}"
                    frame_path = save_frames / frame_filename
                    cv2.imwrite(str(frame_path), frame_info.frame)

                # Display frame
                cv2.imshow("Valoscribe Frame Viewer", frame_info.frame)

                # Add frame info overlay
                info_text = (
                    f"Frame: {frame_info.frame_number} | "
                    f"Time: {frame_info.timestamp_sec:.2f}s | "
                    f"Total processed: {frame_count + 1}"
                )
                typer.echo(f"\r{info_text}", nl=False)

                # Handle keyboard input
                key = cv2.waitKey(1 if not paused else 0) & 0xFF

                if key == ord('q'):
                    typer.echo("\n\nQuitting...")
                    break
                elif key == ord(' '):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    typer.echo(f"\n{status}")

                frame_count += 1

            typer.echo(f"\n\nProcessed {frame_count} frames")

            if save_frames:
                typer.secho(f"Frames saved to: {save_frames}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    finally:
        cv2.destroyAllWindows()


@app.command()
def crop(
    video_path: Path = typer.Argument(..., help="Path to the video file to read"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    save_crops: Optional[Path] = typer.Option(
        None,
        "--save-crops",
        "-s",
        help="Directory to save cropped regions (if not specified, only displayed)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        None,
        "--fps",
        "-f",
        help="Process frames at this FPS (e.g., 5 for 5 frames/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process (for debugging)",
    ),
    crop_format: str = typer.Option(
        "jpg",
        "--format",
        help="Crop image format when saving (jpg, png, etc.)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging to see frame seeking details",
    ),
) -> None:
    """Display and save cropped HUD regions from a Valorant VOD."""
    import logging
    setup_logging(level=logging.DEBUG if debug else logging.INFO)

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")

            if fps:
                typer.echo(f"Processing at: {fps} fps")
            if start_time:
                typer.echo(f"Start time: {start_time}s")
            if end_time:
                typer.echo(f"End time: {end_time}s")
            if max_frames:
                typer.echo(f"Max frames: {max_frames}")

            if save_crops:
                save_crops.mkdir(parents=True, exist_ok=True)
                typer.echo(f"Saving crops to: {save_crops}")

            typer.echo("\nPress 'q' to quit, SPACE to pause/resume\n")

            frame_count = 0
            paused = False

            for frame_info in reader:
                # Check max_frames limit
                if max_frames and frame_count >= max_frames:
                    typer.echo(f"\nReached max frames limit: {max_frames}")
                    break

                # Crop all regions
                cropped_regions = cropper.crop_all_regions(frame_info.frame)

                # Save crops if requested
                if save_crops:
                    frame_dir = save_crops / f"frame_{frame_info.frame_number:06d}"
                    frame_dir.mkdir(exist_ok=True)

                    # Save simple regions
                    for region_name in ["round_number", "team1_score", "team2_score", "round_timer", "minimap"]:
                        crop = cropped_regions[region_name]
                        filename = f"{region_name}.{crop_format}"
                        cv2.imwrite(str(frame_dir / filename), crop)

                    # Save killfeed
                    for i, kill_crop in enumerate(cropped_regions["killfeed"]):
                        filename = f"killfeed_{i}.{crop_format}"
                        cv2.imwrite(str(frame_dir / filename), kill_crop)

                    # Save full player boxes (before subdivision)
                    player_info = cropper.regions["player_info"]
                    individual_height = player_info["individual_height"]
                    offset = player_info["offset"]

                    # Left side (players 0-4)
                    for i in range(5):
                        y_start = player_info["y"] + i * (individual_height + offset)
                        y_end = y_start + individual_height
                        x_start = player_info["x"]
                        x_end = x_start + player_info["width"]

                        full_crop = frame_info.frame[y_start:y_end, x_start:x_end]
                        if full_crop.size > 0:
                            filename = f"player{i}_full.{crop_format}"
                            cv2.imwrite(str(frame_dir / filename), full_crop)

                    # Right side (players 5-9) - need to mirror
                    mirrored_frame = cv2.flip(frame_info.frame, 1)
                    for i in range(5):
                        y_start = player_info["y"] + i * (individual_height + offset)
                        y_end = y_start + individual_height
                        x_start = player_info["x"]
                        x_end = x_start + player_info["width"]

                        full_crop_mirrored = mirrored_frame[y_start:y_end, x_start:x_end]
                        # Flip back for correct orientation
                        full_crop = cv2.flip(full_crop_mirrored, 1)
                        if full_crop.size > 0:
                            filename = f"player{i+5}_full.{crop_format}"
                            cv2.imwrite(str(frame_dir / filename), full_crop)

                    # Save player info elements
                    for i, player_crops in enumerate(cropped_regions["player_info"]):
                        for element_name, element_crop in player_crops.items():
                            if element_name != "side" and element_crop.size > 0:
                                filename = f"player{i}_{element_name}.{crop_format}"
                                cv2.imwrite(str(frame_dir / filename), element_crop)

                    # Save full pre-round player boxes
                    player_info_preround = cropper.regions["player_info_preround"]
                    individual_height_preround = player_info_preround["individual_height"]
                    offset_preround = player_info_preround["offset"]

                    # Left side pre-round (players 0-4)
                    for i in range(5):
                        y_start = player_info_preround["y"] + i * (individual_height_preround + offset_preround)
                        y_end = y_start + individual_height_preround
                        x_start = player_info_preround["x"]
                        x_end = x_start + player_info_preround["width"]

                        full_crop = frame_info.frame[y_start:y_end, x_start:x_end]
                        if full_crop.size > 0:
                            filename = f"player{i}_preround_full.{crop_format}"
                            cv2.imwrite(str(frame_dir / filename), full_crop)

                    # Right side pre-round (players 5-9) - need to mirror
                    for i in range(5):
                        y_start = player_info_preround["y"] + i * (individual_height_preround + offset_preround)
                        y_end = y_start + individual_height_preround
                        x_start = player_info_preround["x"]
                        x_end = x_start + player_info_preround["width"]

                        full_crop_mirrored = mirrored_frame[y_start:y_end, x_start:x_end]
                        # Flip back for correct orientation
                        full_crop = cv2.flip(full_crop_mirrored, 1)
                        if full_crop.size > 0:
                            filename = f"player{i+5}_preround_full.{crop_format}"
                            cv2.imwrite(str(frame_dir / filename), full_crop)

                    # Save pre-round player info elements
                    player_crops_preround = cropper.crop_player_info_preround(frame_info.frame)
                    for i, player_crops in enumerate(player_crops_preround):
                        for element_name, element_crop in player_crops.items():
                            if element_name != "side" and element_crop.size > 0:
                                filename = f"player{i}_preround_{element_name}.{crop_format}"
                                cv2.imwrite(str(frame_dir / filename), element_crop)

                # Display full frame
                cv2.imshow("Full Frame", frame_info.frame)

                # Display simple regions in a grid
                simple_display = []
                for region_name in ["round_number", "team1_score", "team2_score", "round_timer"]:
                    crop = cropped_regions[region_name]
                    if crop.size > 0:
                        # Resize for better visibility
                        resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                        simple_display.append(resized)

                # Stack simple regions horizontally if they exist
                if simple_display:
                    # Pad to same height
                    max_height = max(img.shape[0] for img in simple_display)
                    padded = []
                    for img in simple_display:
                        if img.shape[0] < max_height:
                            pad = max_height - img.shape[0]
                            img = cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        padded.append(img)
                    simple_grid = np.hstack(padded)
                    cv2.imshow("Simple Regions", simple_grid)

                # Display minimap
                if cropped_regions["minimap"].size > 0:
                    cv2.imshow("Minimap", cropped_regions["minimap"])

                # Display first few killfeed entries
                for i, kill_crop in enumerate(cropped_regions["killfeed"][:5]):
                    if kill_crop.size > 0:
                        cv2.imshow(f"Killfeed {i}", kill_crop)

                # Display first 2 players' info
                for i in range(min(2, len(cropped_regions["player_info"]))):
                    player_crops = cropped_regions["player_info"][i]

                    # Show abilities
                    ability_crops = []
                    for ability_name in ["ability_1", "ability_2", "ability_3"]:
                        crop = player_crops.get(ability_name, np.array([]))
                        if crop.size > 0:
                            # Resize for visibility
                            resized = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                            ability_crops.append(resized)

                    if ability_crops:
                        # Pad to same height
                        max_h = max(c.shape[0] for c in ability_crops)
                        padded = []
                        for c in ability_crops:
                            if c.shape[0] < max_h:
                                pad = max_h - c.shape[0]
                                c = cv2.copyMakeBorder(c, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            padded.append(c)
                        abilities_display = np.hstack(padded)
                        cv2.imshow(f"Player {i} Abilities", abilities_display)

                    # Show ultimate
                    ult_crop = player_crops.get("ultimate", np.array([]))
                    if ult_crop.size > 0:
                        ult_resized = cv2.resize(ult_crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                        cv2.imshow(f"Player {i} Ultimate", ult_resized)

                # Status update
                info_text = (
                    f"Frame: {frame_info.frame_number} | "
                    f"Time: {frame_info.timestamp_sec:.2f}s | "
                    f"Processed: {frame_count + 1}"
                )
                typer.echo(f"\r{info_text}", nl=False)

                # Handle keyboard input
                key = cv2.waitKey(1 if not paused else 0) & 0xFF

                if key == ord('q'):
                    typer.echo("\n\nQuitting...")
                    break
                elif key == ord(' '):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    typer.echo(f"\n{status}")

                frame_count += 1

            typer.echo(f"\n\nProcessed {frame_count} frames")

            if save_crops:
                typer.secho(f"Crops saved to: {save_crops}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    finally:
        cv2.destroyAllWindows()


@app.command()
def process(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to coordinate mapping config file",
    ),
    fps: int = typer.Option(5, "--fps", "-f", help="Frames per second to process"),
    output: Path = typer.Option(
        Path("./output/events.json"),
        "--output",
        "-o",
        help="Output path for event timeline",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose event output",
    ),
) -> None:
    """Process a Valorant VOD and extract event timeline."""
    typer.echo(f"Processing video: {video_path}")
    typer.echo(f"FPS: {fps}")
    typer.echo(f"Verbose: {verbose}")
    # TODO: Implement video processing pipeline
    typer.secho("Processing functionality not yet implemented", fg=typer.colors.YELLOW)


@app.command()
def analyze(
    events_path: Path = typer.Argument(..., help="Path to events JSON file"),
) -> None:
    """Analyze extracted events (placeholder for future EDA/ML)."""
    typer.echo(f"Analyzing events from: {events_path}")
    # TODO: Implement analysis logic
    typer.secho("Analysis functionality not yet implemented", fg=typer.colors.YELLOW)


@app.command(name="split-metadata")
def split_metadata(
    series_metadata_path: Path = typer.Argument(
        ..., help="Path to series metadata JSON (from VLR scraper)"
    ),
    output_dir: Path = typer.Option(
        Path("./maps"),
        "--output-dir",
        "-o",
        help="Output directory for individual map metadata files",
    ),
    prefix: str = typer.Option(
        "map",
        "--prefix",
        "-p",
        help="Prefix for output files (e.g., 'map' -> map1.json, map2.json)",
    ),
) -> None:
    """
    Split series metadata into individual map metadata files.

    Takes the VLR scraper output (series-level metadata with multiple maps)
    and creates separate metadata files for each map, compatible with
    GameStateManager.

    Example:
        valoscribe split-metadata series.json --output-dir ./maps/
        # Creates: maps/map1.json, maps/map2.json, etc.
    """
    setup_logging()

    # Load series metadata
    typer.echo(f"Loading series metadata from {series_metadata_path}...")
    try:
        with open(series_metadata_path) as f:
            series_metadata = json.load(f)
    except Exception as e:
        typer.echo(f"Error loading metadata: {e}", err=True)
        raise typer.Exit(1)

    # Validate structure
    if "teams" not in series_metadata or "maps" not in series_metadata:
        typer.echo(
            "Error: Invalid series metadata format (missing 'teams' or 'maps')",
            err=True,
        )
        raise typer.Exit(1)

    team_names = series_metadata["teams"]
    maps = series_metadata["maps"]

    typer.echo(f"Series: {team_names[0]} vs {team_names[1]}")
    typer.echo(f"Maps found: {len(maps)}")
    typer.echo("")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split each map into separate file
    for map_data in maps:
        map_number = map_data.get("map_number", 0)
        map_name = map_data.get("map_name", "Unknown")

        typer.echo(f"Processing Map {map_number}: {map_name}")

        # Extract teams data from map
        map_teams = map_data.get("teams", [])

        # Flatten players from nested team structure to flat list with team field
        all_players = []
        for team in map_teams:
            team_name = team["name"]
            for player in team.get("players", []):
                all_players.append({
                    "name": player["name"],
                    "team": team_name,
                    "agent": player["agent"],
                })

        # Transform to GameStateManager format
        map_metadata = {
            "teams": [
                {
                    "name": team["name"],
                    "starting_side": team["starting_side"],
                }
                for team in map_teams
            ],
            "players": all_players,
            "map": map_name,
            "map_number": map_number,
            "vod_url": map_data.get("vod_url"),
            "match_url": series_metadata.get("match_url"),
        }

        # Write to file
        output_file = output_dir / f"{prefix}{map_number}.json"
        with open(output_file, "w") as f:
            json.dump(map_metadata, f, indent=2)

        typer.echo(f"  ✓ Written to {output_file}")
        typer.echo(f"    VOD: {map_metadata.get('vod_url', 'N/A')}")
        starting_sides = " / ".join([t["starting_side"] for t in map_teams])
        typer.echo(f"    Starting sides: {starting_sides}")
        typer.echo(f"    Players: {len(all_players)}")
        typer.echo("")

    # Summary
    typer.echo("=" * 60)
    typer.echo(f"✓ Split complete! Created {len(maps)} map metadata files")
    typer.echo(f"Output directory: {output_dir}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  1. Download VODs for each map")
    typer.echo("  2. Process with GameStateManager:")
    for map_data in maps:
        map_number = map_data.get("map_number", 0)
        typer.echo(
            f"     valoscribe orchestrate process-vod map{map_number}.mp4 "
            f"{output_dir}/{prefix}{map_number}.json"
        )


@app.command()
def version() -> None:
    """Show valoscribe version."""
    typer.echo("valoscribe v0.1.0")

