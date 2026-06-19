"""
Template-based weapon detector for Valorant HUD.

Matches each player's weapon region against a library of weapon icon templates.
Templates live in src/valoscribe/templates/weapons/<weapon_name>[N].png.

Multiple variants per weapon are supported: vandal1.png, vandal2.png, vandal3.png
all load as variants of "vandal".  Single-image weapons can be named either
frenzy1.png or frenzy.png — both work.

Run scripts/extract_weapon_templates.py once to bootstrap the template library
from processed VOD frames.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import WeaponInfo, WeaponTier
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


WEAPON_TIERS: dict[str, WeaponTier] = {
    # Sidearms
    "classic": WeaponTier.SIDEARM,
    "shorty": WeaponTier.SIDEARM,
    "frenzy": WeaponTier.SIDEARM,
    "ghost": WeaponTier.SIDEARM,
    "sheriff": WeaponTier.SIDEARM,
    "bandit": WeaponTier.SIDEARM,
    # SMGs
    "stinger": WeaponTier.SMG,
    "spectre": WeaponTier.SMG,
    # Shotguns
    "bucky": WeaponTier.SHOTGUN,
    "judge": WeaponTier.SHOTGUN,
    # Rifles
    "bulldog": WeaponTier.RIFLE,
    "guardian": WeaponTier.RIFLE,
    "phantom": WeaponTier.RIFLE,
    "vandal": WeaponTier.RIFLE,
    # Snipers
    "marshal": WeaponTier.SNIPER,
    "outlaw": WeaponTier.SNIPER,
    "operator": WeaponTier.SNIPER,
    # Heavy
    "ares": WeaponTier.HEAVY,
    "odin": WeaponTier.HEAVY,
    # Melee
    "melee": WeaponTier.MELEE,
    "knife": WeaponTier.MELEE,
}


class TemplateWeaponDetector:
    """
    Template matching-based weapon detector.

    Loads all PNG templates from templates/weapons/ and groups them by base
    weapon name.  Variant suffixes are stripped: vandal1.png, vandal2.png, and
    vandal3.png all contribute to the "vandal" entry.  Files without a trailing
    digit (e.g. phantom.png) are also accepted.

    Detection scores every (weapon, variant) pair and returns the base weapon
    name with the highest match score across all variants.
    """

    def __init__(
        self,
        cropper: Cropper,
        template_dir: Optional[Path] = None,
        min_confidence: float = 0.60,
        match_method: int = cv2.TM_CCOEFF_NORMED,
    ):
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.match_method = match_method

        if template_dir is None:
            package_dir = Path(__file__).parent.parent
            template_dir = package_dir / "templates" / "weapons"

        self.template_dir = Path(template_dir)
        # weapon_name -> list of (grayscale, edge) variant pairs
        # Canonical height for normalized comparison.
        # Tight-cropped gun pixels are resized to this height while preserving
        # aspect ratio, so position variance within the HUD strip is removed
        # without distorting the silhouette shape that distinguishes weapons.
        self._canon_h: int = 14
        self.templates: dict[str, list[np.ndarray]] = self._load_templates()

        total_variants = sum(len(v) for v in self.templates.values())
        log.info(
            f"TemplateWeaponDetector: {len(self.templates)} weapons, "
            f"{total_variants} total variant templates loaded "
            f"(min_confidence={min_confidence})"
        )

    def _load_templates(self) -> dict[str, list[np.ndarray]]:
        templates: dict[str, list[np.ndarray]] = {}
        if not self.template_dir.exists():
            log.warning(
                f"Weapon template directory not found: {self.template_dir}. "
                "Run scripts/extract_weapon_templates.py to bootstrap templates."
            )
            return templates

        for path in sorted(self.template_dir.glob("*.png")):
            base_name = re.sub(r"\d+$", "", path.stem).lower()
            if not base_name:
                log.warning(f"Could not determine weapon name from filename: {path.name}")
                continue

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                log.warning(f"Failed to load weapon template: {path}")
                continue

            if base_name not in templates:
                templates[base_name] = []
            normalized = self._normalize_icon(img)
            templates[base_name].append(normalized)
            log.debug(f"Loaded weapon template: {base_name} variant {len(templates[base_name])} {img.shape}")

        if not templates:
            log.warning(
                "No weapon templates found. Weapon detection will return UNKNOWN. "
                "Run scripts/extract_weapon_templates.py to bootstrap templates."
            )
        return templates

    def detect(
        self, frame: np.ndarray, player_index: int, side: str = "left"
    ) -> Optional[WeaponInfo]:
        """
        Detect weapon for a specific player.

        Args:
            frame: Full 1080p frame
            player_index: Player index 0-9
            side: 'left' or 'right' (determines which side's crops to use)

        Returns:
            WeaponInfo, or None on crop/size error.
            weapon_name is 'unknown' and tier is UNKNOWN if no template matches.
        """
        if not self.templates:
            return WeaponInfo(
                weapon_name="unknown",
                weapon_tier=WeaponTier.UNKNOWN,
                confidence=0.0,
            )

        player_crops = self.cropper.crop_player_info(frame)
        if player_index >= len(player_crops):
            return None

        crop_data = player_crops[player_index]
        if crop_data.get("side") != side:
            return None

        weapon_crop = crop_data.get("weapon")
        if weapon_crop is None or weapon_crop.size == 0:
            return None

        gray_crop = self._preprocess(weapon_crop)
        norm_crop = self._normalize_icon(gray_crop)
        if norm_crop is None:
            return WeaponInfo(weapon_name="unknown", weapon_tier=WeaponTier.UNKNOWN, confidence=0.0)

        best_name = "unknown"
        best_conf = 0.0

        for weapon_name, variants in self.templates.items():
            for tmpl_norm in variants:
                if tmpl_norm is None:
                    continue
                # Ensure template fits inside crop; resize template slightly if needed
                th, tw = tmpl_norm.shape[:2]
                ch, cw = norm_crop.shape[:2]
                if tw > cw or th > ch:
                    scale = min(cw / tw, ch / th)
                    tmpl_norm = cv2.resize(tmpl_norm, (max(1, round(tw*scale)), max(1, round(th*scale))), cv2.INTER_AREA)
                result = cv2.matchTemplate(norm_crop, tmpl_norm, self.match_method)
                _, conf, _, _ = cv2.minMaxLoc(result)
                conf = float(max(0.0, min(1.0, conf)))

                if conf > best_conf:
                    best_conf = conf
                    best_name = weapon_name

        if best_conf < self.min_confidence:
            best_name = "unknown"
            tier = WeaponTier.UNKNOWN
        else:
            tier = WEAPON_TIERS.get(best_name, WeaponTier.UNKNOWN)

        log.debug(
            f"Player {player_index} weapon: {best_name} (conf={best_conf:.2f})"
        )

        return WeaponInfo(
            weapon_name=best_name,
            weapon_tier=tier,
            confidence=best_conf,
        )

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        if len(crop.shape) == 3:
            return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return crop.copy()

    def _normalize_icon(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Binarize (Otsu) → tight-crop gun pixels → resize to canon_h preserving
        aspect ratio. Binarization makes both manually-cleaned templates and
        live HUD crops comparable on shape alone, independent of background brightness.
        Returns None if the image is too dark / has no gun pixels.
        """
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cols = np.where(binary.any(axis=0))[0]
        rows = np.where(binary.any(axis=1))[0]
        if len(cols) < 4 or len(rows) < 2:
            return None
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        tight = binary[y0:y1, x0:x1]
        h, w = tight.shape[:2]
        new_w = max(1, round(w * self._canon_h / h))
        return cv2.resize(tight, (new_w, self._canon_h), interpolation=cv2.INTER_AREA)
