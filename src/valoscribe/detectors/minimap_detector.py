"""
Minimap position detector for Valorant HUD.

Uses HSV color filtering to detect attack/defense team dot positions on the
spectator minimap (top-left of screen), then applies a per-map homography
to convert pixel coordinates to normalized [0,1]² map coordinates.

Calibration:
    1. Record 3-4 reference point pairs: minimap pixel (px, py) ↔ known map position.
    2. Run scripts/calibrate_minimap.py to compute the homography matrix.
    3. Store the result in src/valoscribe/config/minimap_homographies.json.

Dot color ranges (HSV):
    These are tuned for standard Champions 2025 broadcast.
    Attack team: pink-red border  H≈177  ~[165-180, 70-220, 70-220]
    Defense team: teal border     H≈79   ~[70-95,  70-200, 150-255]
    Adjust per-broadcast if colors differ for other productions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from collections import Counter

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import MinimapPositions, PlayerPosition
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)

AGENT_TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "active_round_agents"
MINIMAP_ICONS_DIR  = Path(__file__).parent.parent / "templates" / "minimap_icons"

# HSV ranges for Champions 2025 spectator broadcast.
# Calibrated from actual icon RGB values sampled from broadcast frames:
#   Attack icon border: RGB≈(160,70,80)   -> OpenCV HSV H=177, S=143, V=160
#   Defense icon border: RGB≈(145,238,205) -> OpenCV HSV H=79,  S=100, V=238
#
# The attack icon sits at the pink-red end of the hue wheel (H≈177), which wraps
# at OpenCV's boundary (0/180). We OR two narrow sub-ranges to catch both sides.
# Haven's warm brown terrain is H≈10-35 — well away from H=[165-180], so this
# range has far fewer false positives than the old H=[0-20] orange range.
DEFAULT_ATTACK_HSV = (
    np.array([165, 70, 70], dtype=np.uint8),   # H=177 -> H=[165-180]
    np.array([180, 220, 220], dtype=np.uint8),
)
# Wrap-around: H=177->180->0; a few icon pixels land on H=[0-8]
DEFAULT_ATTACK_HSV2 = (
    np.array([0,  70, 70], dtype=np.uint8),
    np.array([8, 220, 220], dtype=np.uint8),
)
DEFAULT_DEFENSE_HSV = (
    np.array([70, 70, 150], dtype=np.uint8),   # H=79 -> H=[70-95]
    np.array([95, 200, 255], dtype=np.uint8),
)

HOMOGRAPHIES_PATH = Path(__file__).parent.parent / "config" / "minimap_homographies.json"
MASKS_DIR = Path(__file__).parent.parent / "config"


class MinimapDetector:
    """
    Detects all 10 player positions from the spectator minimap.

    Steps per frame:
    1. Crop 450×450 minimap region (already configured in champs2025.json).
    2. Convert to HSV.
    3. Color-filter attack icon borders (H≈177 pink-red) and defense borders (H≈79 teal).
    4. Blob-detect centroids for each team (expect up to 5 per team).
    5. Apply homography H[map_name] to convert pixel -> normalized [0,1]² coords.
    """

    # Shape filter for blob detection — applied to both attack and defense.
    # Player icons are roughly circular discs with a small direction indicator.
    # Viper wall appears as a long thin line: high aspect ratio, low circularity.
    # Dead-player X markers: low circularity (~0.2).
    # Terrain patches: irregular, often elongated.
    _ATK_MIN_SOLIDITY = 0.30   # convex-hull-area / contour-area
    _ATK_MAX_ASPECT  = 3.0     # bounding-box w/h  (and 1/h/w > 1/3.0)
    _ATK_MAX_AREA    = 800
    _ATK_MIN_AREA    = 40

    # Defense uses the same shape constraints as attack.
    _DEF_MIN_SOLIDITY = 0.30
    _DEF_MAX_ASPECT  = 3.0
    _DEF_MAX_AREA    = 800
    _DEF_MIN_AREA    = 40

    # Minimum circularity for any accepted blob.
    # Agent icons: ~0.5–0.9 (roughly circular disc).
    # Viper wall (5×100px line): circularity ≈ 0.10 → rejected.
    # Dead-player X marker: circularity ≈ 0.20 → rejected.
    _MIN_CIRCULARITY = 0.30

    # Exported so game_state_manager can use the same threshold for its tracker-aware
    # nearest-neighbour assignment.
    TRACK_MAX_DIST = 0.20

    def __init__(
        self,
        cropper: Cropper,
        map_name: Optional[str] = None,
        homographies_path: Optional[Path] = None,
        attack_hsv: tuple[np.ndarray, np.ndarray] = DEFAULT_ATTACK_HSV,
        attack_hsv2: tuple[np.ndarray, np.ndarray] = DEFAULT_ATTACK_HSV2,
        defense_hsv: tuple[np.ndarray, np.ndarray] = DEFAULT_DEFENSE_HSV,
        min_blob_area: int = 20,
        max_blob_area: int = 2000,
    ):
        """
        Args:
            cropper: Cropper instance (provides minimap crop).
            map_name: Current map name (e.g. "haven"). Used to look up homography.
                      If None, positions are returned in raw pixel coords / minimap [0,1].
            homographies_path: Path to minimap_homographies.json.
            attack_hsv: Primary (lower, upper) HSV bounds for attack icon border.
                        Calibrated for H=177 (pink-red) — use attack_hsv2 for H=0 wrap.
            attack_hsv2: Secondary HSV range (H near 0) OR-ed with attack_hsv.
            defense_hsv: (lower, upper) HSV bounds for defense icon border.
            min_blob_area: Min contour area (px²) for defense team blobs.
            max_blob_area: Max contour area (px²) for defense team blobs.
        """
        self.cropper = cropper
        self.map_name = map_name
        self.attack_hsv = attack_hsv
        self.attack_hsv2 = attack_hsv2
        self.defense_hsv = defense_hsv
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area

        hp = homographies_path or HOMOGRAPHIES_PATH
        self.homographies = self._load_homographies(hp)
        self.H: Optional[np.ndarray] = self._get_homography(map_name)
        self.H_inv: Optional[np.ndarray] = np.linalg.inv(self.H) if self.H is not None else None

        # Per-map binary mask: white = map area, black = UI background to ignore.
        # Loaded from minimap_mask_<map_name>.png in the config directory.
        self._map_mask: Optional[np.ndarray] = self._load_map_mask(map_name)

        # Minimap icon templates: actual icon crops at native scale (~18-21px), with
        # separate attack (pink ring) and defense (teal ring) variants.
        # Regular: {agent}_(atk|def).png  →  _minimap_icon_templates[agent][side]
        # Spike carrier: {agent}_(atk|def)_spike.png  →  _minimap_icon_spike_templates[agent][side]
        self._minimap_icon_templates: dict[str, dict[str, np.ndarray]] = {}
        self._minimap_icon_spike_templates: dict[str, dict[str, np.ndarray]] = {}
        if MINIMAP_ICONS_DIR.exists():
            import re as _re
            for p in MINIMAP_ICONS_DIR.glob("*.png"):
                # Match both "agent_atk" and "agent_atk_spike"
                m_spike = _re.match(r"^(.+)_(atk|def)_spike$", p.stem)
                m_reg   = _re.match(r"^(.+)_(atk|def)$",       p.stem)
                img = cv2.imread(str(p))
                if img is None:
                    continue
                if m_spike:
                    agent, sa = m_spike.group(1), m_spike.group(2)
                    self._minimap_icon_spike_templates.setdefault(agent, {})[sa] = img
                elif m_reg:
                    agent, sa = m_reg.group(1), m_reg.group(2)
                    self._minimap_icon_templates.setdefault(agent, {})[sa] = img
        log.info(
            f"Loaded minimap icon templates for {len(self._minimap_icon_templates)} agents "
            f"({'atk+def' if all('atk' in v and 'def' in v for v in self._minimap_icon_templates.values()) else 'partial'})"
            + (f"; {len(self._minimap_icon_spike_templates)} with spike variant" if self._minimap_icon_spike_templates else "")
        )

        # Fallback portrait templates for agents not in minimap_icons: resized to 20×20.
        _FALLBACK_SIZE = 20
        self._agent_templates: dict[str, np.ndarray] = {}
        for tmpl_path in AGENT_TEMPLATES_DIR.glob("*.jpg"):
            agent = tmpl_path.stem
            if agent in self._minimap_icon_templates:
                continue  # prefer minimap_icons variant
            img = cv2.imread(str(tmpl_path))
            if img is not None:
                self._agent_templates[agent] = cv2.resize(
                    img, (_FALLBACK_SIZE, _FALLBACK_SIZE), interpolation=cv2.INTER_AREA,
                )
        if self._agent_templates:
            log.debug(
                f"Loaded {len(self._agent_templates)} fallback portrait templates "
                f"(agents not in minimap_icons): {sorted(self._agent_templates)}"
            )

        log.info(
            f"MinimapDetector initialized (map={map_name}, "
            f"homography={'loaded' if self.H is not None else 'missing — will use raw pixel coords'})"
        )

    # ── Homography ────────────────────────────────────────────────────────────

    def _load_map_mask(self, map_name: Optional[str]) -> Optional[np.ndarray]:
        """
        Load per-map binary mask from minimap_mask_<map_name>.png.

        The PNG uses the alpha channel: alpha=0 → map area (keep), alpha=255 → background (block).
        Returns a single-channel uint8 array (255 = map, 0 = background), or None if not found.
        """
        if not map_name:
            return None
        path = MASKS_DIR / f"minimap_mask_{map_name}.png"
        if not path.exists():
            log.debug(f"No minimap mask for map '{map_name}' ({path})")
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim < 3:
            log.warning(f"Could not load minimap mask: {path}")
            return None
        if img.shape[2] == 4:
            # alpha=0 → transparent → map area → keep (255); alpha=255 → background → block (0)
            mask = (img[:, :, 3] == 0).astype(np.uint8) * 255
        else:
            # Greyscale or BGR without alpha: treat bright pixels as map area
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        log.info(f"Loaded minimap mask for '{map_name}': {mask.shape}, "
                 f"{(mask==255).sum()} map pixels / {mask.size} total")
        return mask

    def _load_homographies(self, path: Path) -> dict[str, list]:
        if not path.exists():
            log.warning(
                f"Homography file not found: {path}. "
                "Run scripts/calibrate_minimap.py to generate it."
            )
            return {}
        with open(path) as f:
            return json.load(f)

    def _get_homography(self, map_name: Optional[str]) -> Optional[np.ndarray]:
        if not map_name or map_name not in self.homographies:
            return None
        H_list = self.homographies[map_name]
        return np.array(H_list, dtype=np.float64)

    def set_map(self, map_name: str) -> None:
        """Update the active map, reload homography and mask, and clear position tracks."""
        self.map_name = map_name
        self.H = self._get_homography(map_name)
        self._map_mask = self._load_map_mask(map_name)
        if self.H is None:
            log.warning(
                f"No homography for map '{map_name}'. Positions will be raw minimap coords."
            )

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        alive_atk: int = 5,
        alive_def: int = 5,
    ) -> MinimapPositions:
        """
        Detect raw player blob positions from the minimap.

        Blobs are ranked by circularity (most circular = most likely a live player icon)
        and capped at alive_atk / alive_def.  Identity tracking (which blob = which player)
        is NOT done here — the caller (game_state_manager._distribute_minimap_positions)
        matches blobs to specific player trackers via nearest-neighbour on each tracker's
        last-known position.

        Args:
            alive_atk: Cap on attack blobs returned.
            alive_def: Cap on defense blobs returned.

        Returns MinimapPositions with raw blob positions, up to alive_N per side.
        """
        minimap = self.cropper.crop_simple_region(frame, "minimap")
        if minimap.size == 0:
            return MinimapPositions()

        minimap_hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        minimap_h, minimap_w = minimap.shape[:2]

        atk_centroids = self._find_centroids(
            minimap_hsv, self.attack_hsv, minimap.shape,
            hsv_range2=self.attack_hsv2,
            min_area=self._ATK_MIN_AREA,
            max_area=self._ATK_MAX_AREA,
            min_solidity=self._ATK_MIN_SOLIDITY,
            max_aspect=self._ATK_MAX_ASPECT,
            min_circularity=self._MIN_CIRCULARITY,
            n_max=alive_atk,
        )
        def_centroids = self._find_centroids(
            minimap_hsv, self.defense_hsv, minimap.shape, morph_open=True,
            min_area=self._DEF_MIN_AREA,
            max_area=self._DEF_MAX_AREA,
            min_solidity=self._DEF_MIN_SOLIDITY,
            max_aspect=self._DEF_MAX_ASPECT,
            min_circularity=self._MIN_CIRCULARITY,
            n_max=alive_def,
        )

        log.debug(
            f"Minimap: {len(atk_centroids)} attack blobs, {len(def_centroids)} defense blobs"
        )

        atk_positions = [
            PlayerPosition(x=nx, y=ny, confidence=1.0)
            for nx, ny in self._pixels_to_norm(atk_centroids, minimap_w, minimap_h)
        ]
        def_positions = [
            PlayerPosition(x=nx, y=ny, confidence=1.0)
            for nx, ny in self._pixels_to_norm(def_centroids, minimap_w, minimap_h)
        ]

        return MinimapPositions(
            attack_positions=atk_positions,
            defense_positions=def_positions,
            atk_detected=len(atk_positions),
            def_detected=len(def_positions),
        )

    def get_masked_minimap(
        self,
        frame: np.ndarray,
        size: tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        """
        Return the minimap cropped, background-masked, and downscaled.

        Background pixels (outside the playable map area) are set to black so the
        CNN branch sees only the map content — the same technique valopreds uses
        with their bw.png overlay.  Size defaults to 128×128: player dots (~8-12px
        in the original 450×450 crop) resolve to 2-4px at this scale, preserving
        enough spatial detail for the CNN without significant compute cost.

        Returns:
            uint8 BGR array of shape (size[1], size[0], 3).
        """
        minimap = self.cropper.crop_simple_region(frame, "minimap")
        if minimap.size == 0:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        if self._map_mask is not None:
            masked = minimap.copy()
            masked[self._map_mask == 0] = 0
        else:
            masked = minimap

        return cv2.resize(masked, size, interpolation=cv2.INTER_AREA)

    def identify_agents(
        self,
        frame: np.ndarray,
        roster: list[tuple[str, str]],
        min_score: float = 0.45,
        search_radius: int = 24,
    ) -> dict[tuple[str, str], tuple[float, float]]:
        """
        Identify which agent each minimap blob belongs to using blob-centric
        Hungarian assignment.

        Uses side-specific minimap icon templates (attack=pink ring, defense=teal ring)
        at their native scale (~18-21px) so the ring color and portrait both contribute
        to the match score.  Falls back to rescaled portrait templates for agents not
        in the minimap_icons set.

        Each side is solved independently, so duplicate agents (sova on both teams)
        never conflict.

        Args:
            frame: Full game frame (minimap is cropped internally).
            roster: List of (agent_name, side) for ALIVE players only.
            min_score: TM_CCOEFF_NORMED floor; assignments below this are dropped.
                       Native-scale icons matching correctly score 0.6+; wrong matches
                       and partial occlusions score lower.
            search_radius: Half-width of the search window around each blob centroid (px).
                           48px window easily contains a ~20px icon with positional slack.

        Returns:
            Dict mapping (agent_name, side) -> (norm_x, norm_y).
        """
        from scipy.optimize import linear_sum_assignment

        if not roster:
            return {}
        if not self._minimap_icon_templates and not self._agent_templates:
            return {}

        minimap = self.cropper.crop_simple_region(frame, "minimap")
        if minimap.size == 0:
            return {}
        mm_h, mm_w = minimap.shape[:2]

        # Cap blob detection to alive players per side — dead players are known
        # from the killfeed and excluded from the roster before this call.
        alive_atk = sum(1 for _, s in roster if s == "attack")
        alive_def = sum(1 for _, s in roster if s == "defense")
        mm_pos = self.detect(frame, alive_atk=alive_atk, alive_def=alive_def)

        results: dict[tuple[str, str], tuple[float, float]] = {}

        for side in ("attack", "defense"):
            side_abbrev = "atk" if side == "attack" else "def"
            blobs = list(
                mm_pos.attack_positions if side == "attack" else mm_pos.defense_positions
            )
            agents = [agent for agent, s in roster if s == side]

            if not blobs or not agents:
                continue

            n_a, n_b = len(agents), len(blobs)

            # Build score matrix: score_mat[agent_idx, blob_idx]
            score_mat = np.zeros((n_a, n_b), dtype=np.float32)
            for ai, agent in enumerate(agents):
                # Prefer side-specific minimap icon; fall back to rescaled portrait.
                icon_variants = self._minimap_icon_templates.get(agent, {})
                tmpl = icon_variants.get(side_abbrev)
                if tmpl is None:
                    tmpl = icon_variants.get("atk" if side_abbrev == "def" else "def")
                if tmpl is None:
                    tmpl = self._agent_templates.get(agent)
                if tmpl is None:
                    log.debug(f"No template for agent '{agent}' — skipping in identity matching")
                    continue

                ts_h, ts_w = tmpl.shape[:2]

                for bi, blob in enumerate(blobs):
                    if self.H_inv is not None:
                        pt = np.array([[[blob.x, blob.y]]], dtype=np.float64)
                        px_pt = cv2.perspectiveTransform(pt, self.H_inv)
                        cx = int(round(float(px_pt[0, 0, 0])))
                        cy = int(round(float(px_pt[0, 0, 1])))
                    else:
                        cx = int(blob.x * mm_w)
                        cy = int(blob.y * mm_h)
                    x0 = max(0, cx - search_radius)
                    y0 = max(0, cy - search_radius)
                    x1 = min(mm_w, cx + search_radius)
                    y1 = min(mm_h, cy + search_radius)
                    crop = minimap[y0:y1, x0:x1]
                    if crop.shape[0] < ts_h or crop.shape[1] < ts_w:
                        continue
                    score_mat[ai, bi] = float(cv2.minMaxLoc(cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED))[1])

            # Hungarian: minimise cost = maximise score
            row_ind, col_ind = linear_sum_assignment(1.0 - score_mat)

            for ai, bi in zip(row_ind, col_ind):
                score = score_mat[ai, bi]
                agent = agents[ai]
                if score < min_score:
                    log.debug(
                        f"[IDENTITY] {agent}/{side} rejected (score={score:.3f} < {min_score})"
                    )
                    continue
                blob = blobs[bi]
                log.debug(f"[IDENTITY] {agent}/{side} matched blob {bi} (score={score:.3f})")
                results[(agent, side)] = (round(blob.x, 4), round(blob.y, 4))

        # Spike fallback for identify_agents: retry unassigned attack players with spike templates.
        if self._minimap_icon_spike_templates:
            atk_blobs = list(mm_pos.attack_positions)
            unassigned_atk = [a for a, s in roster if s == "attack" and (a, "attack") not in results]
            if unassigned_atk and atk_blobs:
                n_a, n_b = len(unassigned_atk), len(atk_blobs)
                score_mat = np.zeros((n_a, n_b), dtype=np.float32)
                for ai, agent in enumerate(unassigned_atk):
                    spike_t = self._minimap_icon_spike_templates.get(agent, {}).get("atk")
                    if spike_t is None:
                        continue
                    ts_h, ts_w = spike_t.shape[:2]
                    for bi, blob in enumerate(atk_blobs):
                        if self.H_inv is not None:
                            pt = np.array([[[blob.x, blob.y]]], dtype=np.float64)
                            px_pt = cv2.perspectiveTransform(pt, self.H_inv)
                            cx = int(round(float(px_pt[0, 0, 0])))
                            cy = int(round(float(px_pt[0, 0, 1])))
                        else:
                            cx = int(blob.x * mm_w)
                            cy = int(blob.y * mm_h)
                        x0 = max(0, cx - search_radius); x1 = min(mm_w, cx + search_radius)
                        y0 = max(0, cy - search_radius); y1 = min(mm_h, cy + search_radius)
                        crop = minimap[y0:y1, x0:x1]
                        if crop.shape[0] < ts_h or crop.shape[1] < ts_w:
                            continue
                        score_mat[ai, bi] = float(cv2.minMaxLoc(cv2.matchTemplate(crop, spike_t, cv2.TM_CCOEFF_NORMED))[1])
                row_ind, col_ind = linear_sum_assignment(1.0 - score_mat)
                for ai, bi in zip(row_ind, col_ind):
                    score = score_mat[ai, bi]
                    agent = unassigned_atk[ai]
                    if score < min_score:
                        continue
                    blob = atk_blobs[bi]
                    log.debug(f"[IDENTITY] {agent}/attack spike-fallback blob {bi} (score={score:.3f})")
                    results[(agent, "attack")] = (round(blob.x, 4), round(blob.y, 4))

        return results

    def locate_players(
        self,
        frame: np.ndarray,
        roster: list[tuple[str, str]],
        min_score: float = 0.45,
    ) -> dict[tuple[str, str], tuple[float, float, float]]:
        """
        For each alive player in roster, find their position via Hungarian assignment:
          1. Detect ring-colored blob centroids per side (actual icon locations).
          2. Run matchTemplate per agent → full heatmap.
          3. Sample each heatmap at every blob centroid → score matrix (agents × blobs).
          4. scipy linear_sum_assignment for optimal one-to-one assignment.

        This avoids the independent-global-max problem where multiple agents claim
        the same dot and each agent's template is matched to the wrong icon.

        Falls back to independent global-max if no ring-colored blobs are detected
        (smoke-covered or partially-visible minimap).

        Args:
            frame: Full 1080p game frame.
            roster: [(agent_name, side), ...] for ALIVE players only.
            min_score: Acceptance floor after assignment.

        Returns:
            {(agent, side): (norm_x, norm_y, confidence)}
        """
        from scipy.optimize import linear_sum_assignment as _lsa

        minimap = self.cropper.crop_simple_region(frame, "minimap")
        if minimap.size == 0:
            return {}

        mm_h, mm_w = minimap.shape[:2]
        minimap_hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        _search_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (28, 28))

        # Loose ring masks — for zeroing non-ring regions before template search.
        # Catches rings even with slight observer colour shift.
        atk_loose = cv2.bitwise_or(
            cv2.inRange(minimap_hsv,
                        np.array([165, 70, 70], dtype=np.uint8),
                        np.array([180, 220, 220], dtype=np.uint8)),
            cv2.inRange(minimap_hsv,
                        np.array([0,  70, 70], dtype=np.uint8),
                        np.array([8, 220, 220], dtype=np.uint8)),
        )
        def_loose = cv2.inRange(minimap_hsv,
                                np.array([70, 70, 150], dtype=np.uint8),
                                np.array([95, 200, 255], dtype=np.uint8))

        # Tight ring masks — high saturation requirement filters map terrain that
        # shares a similar hue (e.g. Pearl's pinkish floor tiles).  Used only to
        # constrain which regions can contribute NMS candidate positions.
        atk_tight = cv2.bitwise_or(
            cv2.inRange(minimap_hsv,
                        np.array([165, 130, 100], dtype=np.uint8),
                        np.array([180, 255, 255], dtype=np.uint8)),
            cv2.inRange(minimap_hsv,
                        np.array([0,  130, 100], dtype=np.uint8),
                        np.array([8, 255, 255], dtype=np.uint8)),
        )
        def_tight = cv2.inRange(minimap_hsv,
                                np.array([70, 100, 160], dtype=np.uint8),
                                np.array([95, 255, 255], dtype=np.uint8))

        map_mask_resized: Optional[np.ndarray] = None
        if self._map_mask is not None:
            map_mask_resized = self._map_mask
            if map_mask_resized.shape[:2] != (mm_h, mm_w):
                map_mask_resized = cv2.resize(
                    map_mask_resized, (mm_w, mm_h), interpolation=cv2.INTER_NEAREST
                )
            atk_loose = cv2.bitwise_and(atk_loose, map_mask_resized)
            def_loose = cv2.bitwise_and(def_loose, map_mask_resized)
            atk_tight = cv2.bitwise_and(atk_tight, map_mask_resized)
            def_tight = cv2.bitwise_and(def_tight, map_mask_resized)

        atk_ring = cv2.dilate(atk_loose, _search_k)
        def_ring = cv2.dilate(def_loose, _search_k)
        if map_mask_resized is not None:
            atk_ring = cv2.bitwise_and(atk_ring, map_mask_resized)
            def_ring = cv2.bitwise_and(def_ring, map_mask_resized)

        search_atk = minimap.copy(); search_atk[atk_ring == 0] = 0
        search_def = minimap.copy(); search_def[def_ring == 0] = 0

        results: dict[tuple[str, str], tuple[float, float, float]] = {}

        def _resolve_template(agent: str, sa: str):
            iv = self._minimap_icon_templates.get(agent, {})
            t = iv.get(sa)
            if t is None:
                t = iv.get("def" if sa == "atk" else "atk")
            if t is None:
                t = self._agent_templates.get(agent)
            return t

        def _nms_peaks_near_ring(
            heat: np.ndarray,
            n: int,
            suppress_r: int,
            ring_mask: np.ndarray,
            ts_w: int,
            ts_h: int,
        ) -> list[tuple[int, int]]:
            """Top-n peaks in heatmap that land near a vivid ring-coloured pixel.

            Only peaks whose template-centre falls within suppress_r pixels of any
            set pixel in ring_mask are accepted.  Peaks that don't land near a ring
            are suppressed and skipped — no fallback to unconstrained peaks.  Fewer
            detections is preferable to phantom detections on map terrain.
            """
            h_copy = heat.copy()
            peaks: list[tuple[int, int]] = []

            for _ in range(n * 3):  # over-sample to fill quota despite filter
                _, max_val, _, max_loc = cv2.minMaxLoc(h_copy)
                if max_val <= 0.0:
                    break
                rx, ry = max_loc
                # Template centre in mask coordinates
                cx = int(np.clip(rx + ts_w / 2, 0, ring_mask.shape[1] - 1))
                cy = int(np.clip(ry + ts_h / 2, 0, ring_mask.shape[0] - 1))
                x0m = max(0, cx - suppress_r); x1m = min(ring_mask.shape[1], cx + suppress_r + 1)
                y0m = max(0, cy - suppress_r); y1m = min(ring_mask.shape[0], cy + suppress_r + 1)
                if np.any(ring_mask[y0m:y1m, x0m:x1m]):
                    peaks.append((rx, ry))
                # Suppress this region unconditionally
                x0 = max(0, rx - suppress_r); x1 = min(h_copy.shape[1], rx + suppress_r + 1)
                y0 = max(0, ry - suppress_r); y1 = min(h_copy.shape[0], ry + suppress_r + 1)
                h_copy[y0:y1, x0:x1] = -1.0
                if len(peaks) >= n:
                    break

            return peaks[:n]

        def _assign_side(
            agents: list[str],
            side: str,
            search_img: np.ndarray,
            tight_mask: np.ndarray,
            spike_pass: bool = False,
        ) -> None:
            if not agents:
                return
            sa = "atk" if side == "attack" else "def"

            valid: list[tuple[str, np.ndarray, int, int]] = []
            for agent in agents:
                if spike_pass:
                    tmpl = self._minimap_icon_spike_templates.get(agent, {}).get(sa)
                else:
                    tmpl = _resolve_template(agent, sa)
                if tmpl is None:
                    log.debug(f"[LOCATE] No template for {agent}/{side} (spike_pass={spike_pass})")
                    continue
                ts_h, ts_w = tmpl.shape[:2]
                if search_img.shape[0] < ts_h or search_img.shape[1] < ts_w:
                    continue
                heat = cv2.matchTemplate(search_img, tmpl, cv2.TM_CCOEFF_NORMED)
                valid.append((agent, heat, ts_h, ts_w))

            if not valid:
                return

            nv = len(valid)
            avg_ts_w = int(np.mean([ts_w for _, _, _, ts_w in valid]))
            avg_ts_h = int(np.mean([ts_h for _, _, ts_h, _ in valid]))
            suppress_r = max(avg_ts_w // 2, 8)

            # Resize all heatmaps to the same shape so that NMS candidate
            # coordinates (from max_heat) and score_mat lookups use the
            # same coordinate space.  Without this, a ±1px template-size
            # difference causes a coordinate mismatch that drops strong peaks.
            ref_h, ref_w = valid[0][1].shape[:2]
            heat_norm: list[np.ndarray] = []
            for _, heat, _, _ in valid:
                h_f = heat.astype(np.float64)
                if h_f.shape[:2] != (ref_h, ref_w):
                    h_f = cv2.resize(h_f, (ref_w, ref_h)).astype(np.float64)
                heat_norm.append(h_f)

            max_heat = np.full((ref_h, ref_w), -1.0, dtype=np.float64)
            for h_f in heat_norm:
                np.maximum(max_heat, h_f, out=max_heat)

            candidates = _nms_peaks_near_ring(
                max_heat, nv, suppress_r, tight_mask, avg_ts_w, avg_ts_h
            )
            if not candidates:
                return

            m = len(candidates)
            score_mat = np.full((nv, m), -1.0, dtype=np.float64)

            for ai, h_f in enumerate(heat_norm):
                for ci, (rx, ry) in enumerate(candidates):
                    ry_c = int(np.clip(ry, 0, h_f.shape[0] - 1))
                    rx_c = int(np.clip(rx, 0, h_f.shape[1] - 1))
                    score_mat[ai, ci] = h_f[ry_c, rx_c]

            row_ind, col_ind = _lsa(-score_mat)

            for ai, ci in zip(row_ind, col_ind):
                if ci >= m:
                    continue
                agent, heat, ts_h, ts_w = valid[ai]
                score = float(score_mat[ai, ci])
                rx, ry = candidates[ci]
                # Convert candidate top-left back to icon centre using per-agent template size
                px = float(rx + ts_w / 2)
                py = float(ry + ts_h / 2)
                norm = self._pixels_to_norm([(px, py)], mm_w, mm_h)
                nx, ny = norm[0]
                if score >= min_score:
                    results[(agent, side)] = (nx, ny, score)
                    log.debug(
                        f"[LOCATE] {agent}/{side} -> ({px:.0f},{py:.0f}) "
                        f"world=({nx:.3f},{ny:.3f}) conf={score:.3f}"
                    )
                else:
                    log.debug(
                        f"[LOCATE] {agent}/{side} score={score:.3f} < {min_score}, skipped"
                    )

        atk_agents = [a for a, s in roster if s == "attack"]
        def_agents = [a for a, s in roster if s == "defense"]

        # Dilate tight masks slightly so even fractionally-detected rings seed candidates
        _seed_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        atk_seed = cv2.dilate(atk_tight, _seed_k)
        def_seed = cv2.dilate(def_tight, _seed_k)

        _assign_side(atk_agents, "attack", search_atk, atk_seed)
        _assign_side(def_agents, "defense", search_def, def_seed)

        # Spike fallback: for attack players still unassigned, retry using spike templates.
        if self._minimap_icon_spike_templates:
            unassigned_atk = [a for a in atk_agents if (a, "attack") not in results]
            if unassigned_atk:
                log.debug(f"[LOCATE] Spike fallback for {unassigned_atk}")
                _assign_side(unassigned_atk, "attack", search_atk, atk_seed, spike_pass=True)

        return results

    def _pixels_to_norm(
        self,
        centroids: list[tuple[float, float]],
        minimap_w: int,
        minimap_h: int,
    ) -> list[tuple[float, float]]:
        """Convert pixel centroids to normalized [0,1]² via homography or simple scaling."""
        result = []
        for px, py in centroids:
            if self.H is not None:
                nx, ny = self._apply_homography(px, py)
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
            else:
                nx = px / max(minimap_w, 1)
                ny = py / max(minimap_h, 1)
            result.append((nx, ny))
        return result

    def _find_centroids(
        self,
        minimap_hsv: np.ndarray,
        hsv_range: tuple[np.ndarray, np.ndarray],
        minimap_shape: tuple,
        hsv_range2: Optional[tuple[np.ndarray, np.ndarray]] = None,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        min_solidity: float = 0.0,
        max_aspect: float = 0.0,
        min_circularity: float = 0.0,
        morph_open: bool = False,
        n_max: int = 5,
    ) -> list[tuple[float, float]]:
        """
        Color-filter HSV image and blob-detect dot centroids.

        Args:
            hsv_range2: Optional second range OR-ed with hsv_range (for hue wraparound).
            min_area: Override min blob area; falls back to self.min_blob_area.
            max_area: Override max blob area; falls back to self.max_blob_area.
            min_solidity: Reject blobs with convex-hull fill < this (0=off).
            max_aspect: Reject blobs whose bounding-box w/h exceeds this (0=off).
            min_circularity: Reject blobs with 4π·area/perimeter² < this (0=off).
                             Agent icons score ~0.5–0.9; Viper walls score ~0.10.
            morph_open: Apply MORPH_OPEN after CLOSE to remove noise specks.
            n_max: Max blobs to return. Candidates are ranked by circularity
                   (4π·area/perimeter²) so the most circular blobs — live player icons —
                   win over dead-player X markers and irregular terrain patches.

        Returns list of (px, py) pixel coordinates (within the minimap crop).
        """
        lower, upper = hsv_range
        mask = cv2.inRange(minimap_hsv, lower, upper)
        if hsv_range2 is not None:
            lower2, upper2 = hsv_range2
            mask = cv2.bitwise_or(mask, cv2.inRange(minimap_hsv, lower2, upper2))

        if self._map_mask is not None:
            mm_h, mm_w_local = minimap_shape[:2]
            map_mask = self._map_mask
            if map_mask.shape[:2] != (mm_h, mm_w_local):
                map_mask = cv2.resize(map_mask, (mm_w_local, mm_h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_and(mask, map_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        if morph_open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lo_area = min_area if min_area is not None else self.min_blob_area
        hi_area = max_area if max_area is not None else self.max_blob_area

        candidates = []  # (circularity, cx, cy)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (lo_area <= area <= hi_area):
                continue

            if min_solidity > 0 or max_aspect > 0:
                x, y, w, h = cv2.boundingRect(cnt)
                if max_aspect > 0:
                    asp = w / max(h, 1)
                    if asp > max_aspect or asp < 1.0 / max_aspect:
                        continue
                if min_solidity > 0:
                    hull_area = cv2.contourArea(cv2.convexHull(cnt))
                    if hull_area < 1 or (area / hull_area) < min_solidity:
                        continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            # Circularity: 1.0 for a perfect circle; ~0.10 for Viper wall lines.
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

            if min_circularity > 0 and circularity < min_circularity:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            candidates.append((circularity, float(cx), float(cy)))

        # Take the n_max most-circular blobs (player icons beat X markers and terrain)
        candidates.sort(key=lambda c: -c[0])
        top = candidates[:n_max]

        # Restore stable spatial ordering for tracking consistency
        top.sort(key=lambda c: (c[2], c[1]))
        return [(cx, cy) for _, cx, cy in top]

    def _to_player_positions(
        self,
        centroids: list[tuple[float, float]],
        minimap_w: int,
        minimap_h: int,
        max_players: int = 5,
    ) -> list[Optional[PlayerPosition]]:
        """
        Convert pixel centroids to PlayerPosition objects (padded to max_players).

        If homography H is available, applies it to convert minimap pixel coords
        to normalized map coordinates [0,1]².
        Otherwise, normalizes directly by minimap dimensions.
        """
        positions: list[Optional[PlayerPosition]] = []

        for px, py in centroids[:max_players]:
            if self.H is not None:
                nx, ny = self._apply_homography(px, py)
                # Clamp to [0,1] (homography can extrapolate slightly beyond bounds)
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
            else:
                # Fallback: normalize by minimap size
                nx = px / max(minimap_w, 1)
                ny = py / max(minimap_h, 1)

            positions.append(PlayerPosition(x=nx, y=ny, confidence=1.0))

        # Pad to max_players with None
        while len(positions) < max_players:
            positions.append(None)

        return positions

    def _apply_homography(self, px: float, py: float) -> tuple[float, float]:
        """Apply homography matrix H to a single point."""
        pt = np.array([[[px, py]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])

    # ── Debug ─────────────────────────────────────────────────────────────────

    def visualize(self, frame: np.ndarray, positions: MinimapPositions) -> np.ndarray:
        """
        Draw detected positions on a copy of the minimap for debugging.

        Returns an annotated BGR image of the minimap.
        """
        minimap = self.cropper.crop_simple_region(frame, "minimap").copy()
        h, w = minimap.shape[:2]

        def draw_dots(player_positions, color):
            for pos in player_positions:
                if pos is None:
                    continue
                if self.H is not None:
                    # Inverse transform to get pixel coords for visualization
                    # (only if we have homography; otherwise pos is already normalized to minimap)
                    px = int(pos.x * w)
                    py = int(pos.y * h)
                else:
                    px = int(pos.x * w)
                    py = int(pos.y * h)
                cv2.circle(minimap, (px, py), 6, color, -1)
                cv2.circle(minimap, (px, py), 6, (255, 255, 255), 1)

        draw_dots(positions.attack_positions, (0, 165, 255))   # orange = attack
        draw_dots(positions.defense_positions, (255, 200, 0))  # cyan = defense

        return minimap
