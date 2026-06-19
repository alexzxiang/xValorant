"""
live_capture.py — real-time round win probability from a fullscreen broadcast.

Captures your primary display at 2 fps, feeds every frame through the full
valoscribe GameStateManager (all detectors: health, alive, abilities, ult,
weapon, minimap position), and outputs P(attack wins) for every ACTIVE_ROUND
frame via a small always-on-top overlay window and console.

Prerequisites:
    1. Scrape VLR metadata for the match (only needs team names/agents/sides):
           python scripts/generate_match_metadata.py <vlr_match_id> <map_number>
    2. Have a trained model for the map:
           models/masters_london/<mapname>/best_model.pt
    3. Run:
           python scripts/live_capture.py \\
               --metadata masters_london_2026/nrg_vs_lev/map1_ascent/metadata.json \\
               --model-dir models/masters_london

Options:
    --fps 2            Capture rate (default 2, match training cadence)
    --monitor 1        Which display to capture (1=primary)
    --target-height 1080  Resize height before processing (default 1080)
    --no-overlay       Console output only (no tkinter window)
    --output output/live  Where to write frame_states.csv for debugging

Press Ctrl+C or close the overlay window to stop.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from valoscribe.inference.frame_predictor import FramePredictor
from valoscribe.orchestration.game_state_manager import GameStateManager
from valoscribe.utils.logger import setup_logging
from build_dataset import get_team1_side

try:
    import mss
    _MSS_OK = True
except ImportError:
    _MSS_OK = False

try:
    import tkinter as tk
    _TK_OK = True
except ImportError:
    _TK_OK = False


# ── Overlay window ─────────────────────────────────────────────────────────────

class LiveOverlay:
    """Small always-on-top window: ATK/DEF win probability + round context."""

    WIDTH = 330
    HEIGHT = 175
    MAP_SIZE = 540  # display size of the minimap panel (450 * 1.2)

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Round Win Probability")
        self.root.wm_attributes("-topmost", True)
        self.root.configure(bg="#0d0d0d")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.running = True
        self._swapped = False  # if True, DEF label is on the left
        self._map_open = False
        self._map_win: Optional[tk.Toplevel] = None
        self._map_canvas: Optional[tk.Canvas] = None
        self._map_photo = None  # holds PhotoImage reference to prevent GC
        self._pending_minimap: Optional[tuple] = None

        # Position: top-right corner of primary display
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}+{sw - self.WIDTH - 12}+12")

        # Click-and-drag anywhere on the body to reposition the overlay.
        self._drag_off_x = 0
        self._drag_off_y = 0
        self.root.bind("<Button-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._on_move)

        # Swap + map toggle buttons
        swap_row = tk.Frame(self.root, bg="#0d0d0d")
        swap_row.pack(pady=(4, 0))
        _btn_kw = dict(
            font=("Consolas", 8), fg="#555555", bg="#1a1a1a",
            activeforeground="#aaaaaa", activebackground="#222222",
            relief="flat", bd=0, padx=6, pady=2,
        )
        tk.Button(swap_row, text="swap sides", command=self._toggle_swap, **_btn_kw).pack(side="left")
        tk.Label(swap_row, text="  ", bg="#0d0d0d").pack(side="left")
        self._map_btn = tk.Button(swap_row, text="map", command=self._toggle_map, **_btn_kw)
        self._map_btn.pack(side="left")

        # ATK and DEF on the same row, each with their own colored label
        prob_row = tk.Frame(self.root, bg="#0d0d0d")
        prob_row.pack(pady=(4, 2))

        self._atk_var = tk.StringVar(value="ATK  --%")
        self._atk_lbl = tk.Label(
            prob_row,
            textvariable=self._atk_var,
            font=("Consolas", 17, "bold"),
            fg="#888888", bg="#0d0d0d", padx=6,
        )
        self._atk_lbl.pack(side="left")

        tk.Label(prob_row, text="|", font=("Consolas", 17, "bold"),
                 fg="#333333", bg="#0d0d0d").pack(side="left")

        self._def_var = tk.StringVar(value="DEF  --%")
        self._def_lbl = tk.Label(
            prob_row,
            textvariable=self._def_var,
            font=("Consolas", 17, "bold"),
            fg="#888888", bg="#0d0d0d", padx=6,
        )
        self._def_lbl.pack(side="left")

        # Round / timer row
        self._round_var = tk.StringVar(value="")
        tk.Label(
            self.root,
            textvariable=self._round_var,
            font=("Consolas", 11),
            fg="#999999", bg="#0d0d0d",
        ).pack()

        # Score row
        self._score_var = tk.StringVar(value="Waiting for ACTIVE_ROUND...")
        tk.Label(
            self.root,
            textvariable=self._score_var,
            font=("Consolas", 11),
            fg="#666666", bg="#0d0d0d", pady=6,
        ).pack()

        self._pending: Optional[tuple] = None
        self.root.after(100, self._poll)

    def _start_move(self, event):
        # Record the cursor's offset from the window's top-left at drag start.
        self._drag_off_x = event.x_root - self.root.winfo_x()
        self._drag_off_y = event.y_root - self.root.winfo_y()

    def _on_move(self, event):
        # Use absolute screen coords (x_root/y_root) so the math is correct
        # regardless of which child widget received the event.
        x = event.x_root - self._drag_off_x
        y = event.y_root - self._drag_off_y
        self.root.geometry(f"+{x}+{y}")

    def _toggle_swap(self):
        self._swapped = not self._swapped

    def _toggle_map(self):
        if self._map_open:
            self._map_open = False
            self._map_btn.config(fg="#555555")
            if self._map_win is not None:
                try:
                    self._map_win.destroy()
                except tk.TclError:
                    pass
            self._map_win = None
            self._map_canvas = None
        else:
            self._map_open = True
            self._map_btn.config(fg="#aaaaaa")
            self._open_map_window()

    def _open_map_window(self):
        win = tk.Toplevel(self.root)
        win.title("Minimap")
        win.wm_attributes("-topmost", True)
        win.configure(bg="#0d0d0d")
        win.resizable(False, False)

        def _on_map_close():
            self._map_open = False
            self._map_btn.config(fg="#555555")
            self._map_win = None
            self._map_canvas = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_map_close)

        # Position: just to the left of the main overlay
        sw = self.root.winfo_screenwidth()
        mx = sw - self.WIDTH - self.MAP_SIZE - 24
        win.geometry(f"{self.MAP_SIZE}x{self.MAP_SIZE}+{mx}+12")

        canvas = tk.Canvas(win, width=self.MAP_SIZE, height=self.MAP_SIZE,
                           bg="#0d0d0d", highlightthickness=0)
        canvas.pack()

        # Placeholder text until first frame arrives
        canvas.create_text(
            self.MAP_SIZE // 2, self.MAP_SIZE // 2,
            text="Waiting for ACTIVE_ROUND...",
            fill="#444444", font=("Consolas", 11),
        )

        self._map_win = win
        self._map_canvas = canvas

    def update_minimap(self, minimap_bgr: np.ndarray, player_dots: list):
        """Thread-safe: queue one minimap update."""
        self._pending_minimap = (minimap_bgr, player_dots)

    def _draw_map(self, minimap_bgr: np.ndarray, player_dots: list):
        """Render minimap image + player circles on the map canvas. Must run on main thread."""
        canvas = self._map_canvas
        if canvas is None:
            return

        # BGR numpy → PIL RGB → ImageTk.PhotoImage
        rgb = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_img)
        self._map_photo = photo  # keep reference to prevent GC

        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)

        R = 9  # dot radius in display pixels
        for dx, dy, is_atk, is_alive in player_dots:
            if not is_alive:
                continue
            color = "#ff4455" if is_atk else "#00d4aa"
            canvas.create_oval(
                dx - R, dy - R, dx + R, dy + R,
                outline=color, width=2, fill="",
            )

    def _on_close(self):
        self.running = False
        self.root.destroy()

    def update(
        self,
        prob: float,
        round_num: int,
        timer: float,
        spike_timer: float,
        atk_team: str,
        def_team: str,
        score_atk: int,
        score_def: int,
    ):
        """Thread-safe: queue one update. The main thread applies it on next poll."""
        self._pending = (prob, round_num, timer, spike_timer, atk_team, def_team, score_atk, score_def)

    def _poll(self):
        """Apply any pending update; reschedule."""
        if not self.running:
            return
        if self._pending is not None:
            prob, rn, timer, spike, atk_team, def_team, score_atk, score_def = self._pending
            self._pending = None

            atk = prob * 100
            def_ = (1 - prob) * 100

            def _gradient(p: float) -> str:
                """Smooth red->yellow->green gradient for p in [0,1]."""
                if p <= 0.5:
                    t = p / 0.5
                    r, g, b = 255, int(t * 220), 0
                else:
                    t = (p - 0.5) / 0.5
                    r, g, b = int((1 - t) * 255), int(160 + t * 95), 0
                return f"#{r:02x}{g:02x}{b:02x}"

            # Truncate team names to fit — Consolas 17pt, ~330px wide total
            atk_short = atk_team[:8]
            def_short = def_team[:8]

            if not self._swapped:
                left_text, left_color = f"{atk_short} {atk:.1f}%", _gradient(prob)
                right_text, right_color = f"{def_short} {def_:.1f}%", _gradient(1 - prob)
            else:
                left_text, left_color = f"{def_short} {def_:.1f}%", _gradient(1 - prob)
                right_text, right_color = f"{atk_short} {atk:.1f}%", _gradient(prob)

            self._atk_var.set(left_text)
            self._atk_lbl.config(fg=left_color)

            self._def_var.set(right_text)
            self._def_lbl.config(fg=right_color)

            spike_str = f"  SPIKE {spike:.0f}s" if spike > 0 else ""
            self._round_var.set(f"Round {rn}  |  {timer:.0f}s{spike_str}")
            self._score_var.set(f"{atk_team} (ATK) {score_atk} - {score_def} (DEF) {def_team}")

        if self._pending_minimap is not None and self._map_canvas is not None:
            minimap_bgr, player_dots = self._pending_minimap
            self._pending_minimap = None
            try:
                self._draw_map(minimap_bgr, player_dots)
            except tk.TclError:
                pass

        self.root.after(100, self._poll)

    def run(self):
        self.root.mainloop()


# ── Minimap helper ─────────────────────────────────────────────────────────────

# Minimap region in the 1080p frame (from champs2025.json)
_MM_X, _MM_Y, _MM_W, _MM_H = 10, 20, 450, 450


def _build_minimap_dots(
    manager: GameStateManager,
    frame: np.ndarray,
) -> Optional[tuple[np.ndarray, list]]:
    """
    Crop the minimap from the current frame, convert player positions back to
    minimap pixel coords, and return (minimap_display_bgr, player_dots).

    player_dots: list of (display_x, display_y, is_atk: bool, is_alive: bool)
    Returns None if player trackers are not yet initialized.
    """
    if manager.player_trackers is None:
        return None

    rm = manager.round_manager
    scale = LiveOverlay.MAP_SIZE / _MM_W  # 540 / 450 = 1.2

    # Crop and resize minimap for display
    h, w = frame.shape[:2]
    crop = frame[_MM_Y: min(_MM_Y + _MM_H, h), _MM_X: min(_MM_X + _MM_W, w)].copy()
    if crop.size == 0:
        return None
    minimap_display = cv2.resize(crop, (LiveOverlay.MAP_SIZE, LiveOverlay.MAP_SIZE))

    # Inverse homography: normalized map coords [0,1] → minimap pixel coords [0,450]
    try:
        H_inv = manager.detector_registry.minimap_detector.H_inv
    except AttributeError:
        H_inv = None

    # Which team is currently attacking?
    t1_side = get_team1_side(rm.current_round, rm.starting_sides["team1"])

    player_dots: list[tuple[float, float, bool, bool]] = []
    for tracker in manager.player_trackers:
        px_norm = tracker.current_state.get("pos_x")
        py_norm = tracker.current_state.get("pos_y")
        if px_norm is None or py_norm is None:
            continue

        # Back-project to minimap pixel space
        if H_inv is not None:
            pt = np.array([[[float(px_norm), float(py_norm)]]], dtype=np.float64)
            result = cv2.perspectiveTransform(pt, H_inv)
            mx, my = float(result[0][0][0]), float(result[0][0][1])
        else:
            mx, my = float(px_norm) * _MM_W, float(py_norm) * _MM_H

        team = tracker.metadata.get("team", "")
        if t1_side == "attack":
            is_atk = (team == rm.team_names[0])
        else:
            is_atk = (team == rm.team_names[1])

        is_alive = bool(tracker.current_state.get("alive", True))
        player_dots.append((mx * scale, my * scale, is_atk, is_alive))

    return minimap_display, player_dots


# ── Capture / processing loop ──────────────────────────────────────────────────

def _capture_loop(
    manager: GameStateManager,
    overlay: Optional[LiveOverlay],
    fps: float,
    monitor_idx: int,
    target_height: int,
    stop_event: threading.Event,
):
    """Background thread: grab screen → process_frame → update overlay."""
    if not _MSS_OK:
        print("ERROR: 'mss' is not installed. Run: .venv/Scripts/pip install mss")
        stop_event.set()
        return

    interval = 1.0 / fps
    start_time = time.time()
    log = logging.getLogger(__name__)

    with mss.MSS() as sct:
        if monitor_idx >= len(sct.monitors):
            print(f"ERROR: monitor {monitor_idx} not found. Available: 0-{len(sct.monitors)-1}")
            stop_event.set()
            return

        mon = sct.monitors[monitor_idx]
        print(f"Capturing monitor {monitor_idx}: {mon['width']}x{mon['height']} -> resized to height {target_height}")
        print()

        with manager.output_writer:
            while not stop_event.is_set():
                t0 = time.time()
                timestamp = t0 - start_time

                # Grab frame
                raw = sct.grab(mon)
                frame = np.array(raw)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Resize to target height (valoscribe calibrated for 1080p)
                h, w = frame.shape[:2]
                if h != target_height:
                    new_w = int(w * target_height / h)
                    frame = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

                # Feed into pipeline
                try:
                    manager.process_frame(timestamp, frame)
                except StopIteration:
                    print("Match-end signal received — stopping capture.")
                    stop_event.set()
                    break
                except Exception as exc:
                    log.warning(f"Frame error @ {timestamp:.1f}s: {exc}")

                # Relay latest prediction to overlay
                if manager.frame_predictor is not None:
                    prob = manager.frame_predictor.last_prob
                    if prob is not None and overlay is not None and overlay.running:
                        rm = manager.round_manager
                        timers = manager._last_timers or {}
                        t1_side = get_team1_side(
                            rm.current_round,
                            rm.starting_sides["team1"],
                        )
                        if t1_side == "attack":
                            atk_team, def_team = rm.team_names[0], rm.team_names[1]
                            score_atk = rm.current_score["team1"]
                            score_def = rm.current_score["team2"]
                        else:
                            atk_team, def_team = rm.team_names[1], rm.team_names[0]
                            score_atk = rm.current_score["team2"]
                            score_def = rm.current_score["team1"]
                        overlay.update(
                            prob=prob,
                            round_num=rm.current_round,
                            timer=float(timers.get("game_timer") or 0.0),
                            spike_timer=float(timers.get("spike_timer") or 0.0),
                            atk_team=atk_team,
                            def_team=def_team,
                            score_atk=score_atk,
                            score_def=score_def,
                        )

                # Update minimap panel if it's open
                if overlay is not None and overlay._map_open:
                    result = _build_minimap_dots(manager, frame)
                    if result is not None:
                        overlay.update_minimap(*result)

                # Pace to target fps
                elapsed = time.time() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # Clean up overlay when capture stops
    if overlay is not None and overlay.running:
        try:
            overlay.root.after(0, overlay._on_close)
        except Exception:
            pass


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live round win probability from a fullscreen Valorant broadcast."
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to match metadata.json (from generate_match_metadata.py).",
    )
    parser.add_argument(
        "--model-dir", default="models/masters_london",
        help="Model directory containing per-map checkpoints.",
    )
    parser.add_argument(
        "--output", default="output/live",
        help="Directory for frame_states.csv debug output.",
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Screen capture rate (default: 2.0).",
    )
    parser.add_argument(
        "--monitor", type=int, default=1,
        help="Display index to capture: 1=primary, 2=secondary, … (default: 1).",
    )
    parser.add_argument(
        "--target-height", type=int, default=1080,
        help="Resize frame to this height before processing (default: 1080). "
             "Use 1080 for a 1080p stream; 1440 or 2160 for higher-res monitors.",
    )
    parser.add_argument(
        "--map", default=None,
        help="Map name override (e.g. 'fracture'). Defaults to map in metadata.json.",
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Disable the overlay window; predictions go to console only.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress valoscribe processing logs (predictions still shown).",
    )
    args = parser.parse_args()

    setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    # Always mute the noisy agent detector
    logging.getLogger("valoscribe.detectors.template_agent_detector").setLevel(logging.WARNING)

    # Load metadata
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        sys.exit(f"Metadata not found: {metadata_path}")
    with open(metadata_path, encoding="cp1252") as f:
        vlr_metadata = json.load(f)

    if "teams" not in vlr_metadata or "players" not in vlr_metadata:
        sys.exit("Invalid metadata: missing 'teams' or 'players'.")

    t1 = vlr_metadata["teams"][0]
    t2 = vlr_metadata["teams"][1]
    map_name = (args.map or vlr_metadata.get("map", "")).lower().strip()
    if not map_name:
        sys.exit("--map is required when metadata.json has no 'map' field.")

    print(f"Match:   {t1['name']} ({t1['starting_side']}) vs {t2['name']} ({t2['starting_side']})")
    print(f"Map:     {map_name}")
    print(f"Output:  {args.output}")
    print()

    # Create output dir and GameStateManager
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = GameStateManager(
        video_path=Path("live"),    # dummy — we call process_frame() directly
        vlr_metadata=vlr_metadata,
        output_dir=output_dir,
    )

    # Attach FramePredictor
    fp = FramePredictor(model_dir=Path(args.model_dir), map_name=map_name)
    manager.frame_predictor = fp

    # Build overlay (or skip if --no-overlay / tkinter unavailable)
    use_overlay = _TK_OK and not args.no_overlay
    if not _TK_OK and not args.no_overlay:
        print("Warning: tkinter not available — falling back to console output.")
    overlay = LiveOverlay() if use_overlay else None

    # Run capture in background thread
    stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=_capture_loop,
        args=(manager, overlay, args.fps, args.monitor, args.target_height, stop_event),
        daemon=True,
        name="capture",
    )
    capture_thread.start()

    print(f"Running at {args.fps} fps on monitor {args.monitor}.")
    if use_overlay:
        print("Overlay: top-right corner of your screen.")
    print("Press Ctrl+C (or close overlay) to stop.")
    print("=" * 60)

    try:
        if overlay is not None:
            overlay.run()          # blocks main thread; exits when window closes
        else:
            while not stop_event.is_set():
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        capture_thread.join(timeout=5.0)

    print("Capture stopped.")


if __name__ == "__main__":
    main()
