"""
Killfeed detector for Valorant using agent icon template matching.

Uses color template matching to detect agent kills in the killfeed.
Matches killer icons first, then flipped templates for victim icons.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import KillfeedAgentDetection
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class KillfeedDetector:
    """
    Template matching-based detector for killfeed agent kills.

    Uses color template matching to detect killer and victim agents
    from killfeed entries. Templates represent agents facing right (killer),
    and are flipped horizontally to detect victims (facing left).
    """

    def __init__(
        self,
        cropper: Cropper,
        template_dir: Optional[Path] = None,
        min_confidence: float = 0.75,
        match_method: int = cv2.TM_CCOEFF_NORMED,
        agents: Optional[list[str]] = None,
    ):
        """
        Initialize killfeed detector.

        Args:
            cropper: Cropper instance for extracting killfeed regions
            template_dir: Directory containing agent templates with subdirs 'attack' and 'defense'
                         If None, uses default: src/valoscribe/templates/killfeed_agents/
            min_confidence: Minimum template match confidence (0-1)
            match_method: OpenCV template matching method
            agents: Optional list of agent names to limit matching to. If None, uses all templates.
        """
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.match_method = match_method
        self.agents = agents

        # Set default template directory if not provided
        if template_dir is None:
            package_dir = Path(__file__).parent.parent
            template_dir = package_dir / "templates" / "killfeed_agents"

        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()

        if agents:
            log.info(
                f"Killfeed detector initialized (min_confidence: {min_confidence}, "
                f"templates: {len(self.templates)}, method: {match_method}, "
                f"limited to {len(agents)} agents: {', '.join(agents)})"
            )
        else:
            log.info(
                f"Killfeed detector initialized (min_confidence: {min_confidence}, "
                f"templates: {len(self.templates)}, method: {match_method})"
            )

    def _load_templates(self) -> dict[str, dict]:
        """
        Load agent templates from attack and defense directories.

        If self.agents is set, only loads templates for those agents.

        Expects structure:
        - templates/killfeed_agents/attack/<agent_name>.png (or .jpg)
        - templates/killfeed_agents/defense/<agent_name>.png (or .jpg)

        Returns:
            Dictionary mapping template keys to template data:
            {
                "agent_name_attack": {"image": ndarray, "agent": "agent_name", "side": "attack"},
                "agent_name_defense": {"image": ndarray, "agent": "agent_name", "side": "defense"},
            }
        """
        templates = {}

        if not self.template_dir.exists():
            log.warning(f"Template directory does not exist: {self.template_dir}")
            return templates

        # Load attack templates
        attack_dir = self.template_dir / "attack"
        if attack_dir.exists():
            # Support both .png and .jpg files
            for pattern in ["*.png", "*.jpg"]:
                for template_path in attack_dir.glob(pattern):
                    agent_name = template_path.stem

                    # Filter by agent list if provided
                    if self.agents and agent_name not in self.agents:
                        log.debug(f"Skipping {agent_name} (not in agent list)")
                        continue

                    template = cv2.imread(str(template_path))

                    if template is None:
                        log.warning(f"Failed to load template: {template_path}")
                        continue

                    key = f"{agent_name}_attack"
                    templates[key] = {
                        "image": template,
                        "agent": agent_name,
                        "side": "attack",
                    }
                    log.debug(f"Loaded attack template for {agent_name}: {template.shape}")
        else:
            log.warning(f"Attack template directory not found: {attack_dir}")

        # Load defense templates
        defense_dir = self.template_dir / "defense"
        if defense_dir.exists():
            # Support both .png and .jpg files
            for pattern in ["*.png", "*.jpg"]:
                for template_path in defense_dir.glob(pattern):
                    agent_name = template_path.stem

                    # Filter by agent list if provided
                    if self.agents and agent_name not in self.agents:
                        log.debug(f"Skipping {agent_name} (not in agent list)")
                        continue

                    template = cv2.imread(str(template_path))

                    if template is None:
                        log.warning(f"Failed to load template: {template_path}")
                        continue

                    key = f"{agent_name}_defense"
                    templates[key] = {
                        "image": template,
                        "agent": agent_name,
                        "side": "defense",
                    }
                    log.debug(f"Loaded defense template for {agent_name}: {template.shape}")
        else:
            log.warning(f"Defense template directory not found: {defense_dir}")

        if len(templates) == 0:
            log.error(f"No templates loaded from {self.template_dir}")
        else:
            log.info(f"Loaded {len(templates)} agent templates")

        return templates

    def detect(self, frame: np.ndarray) -> list[Optional[KillfeedAgentDetection]]:
        """
        Detect kills from killfeed entries.

        Processes all entries (0-9) regardless of empty slots, since kills can appear
        in lower slots even when upper slots are empty (stale entries expire first).

        Args:
            frame: Input frame (1080p)

        Returns:
            List of KillfeedAgentDetection for detected kills (None for empty entries)
        """
        # Check if templates are loaded
        if len(self.templates) == 0:
            log.error("No templates loaded, cannot detect killfeed")
            return []

        # Get killfeed entry crops
        killfeed_entries = self.cropper.crop_killfeed(frame)

        # Collect all candidate combinations from all entries
        all_candidates = []

        for entry_idx, entry_crop in enumerate(killfeed_entries):
            if entry_crop.size == 0:
                log.debug(f"Entry {entry_idx}: empty crop, skipping")
                continue  # Skip this entry but continue checking others

            # Get all candidate combinations for this entry (sorted by confidence)
            entry_candidates = self._detect_entry(entry_crop, entry_idx)

            if not entry_candidates:
                # No candidates in this entry, but continue checking remaining entries
                log.debug(f"Entry {entry_idx}: no candidates found")
                continue

            # Add all candidates from this entry
            all_candidates.extend(entry_candidates)

        return all_candidates

    def _detect_entry(
        self, entry_crop: np.ndarray, entry_idx: int
    ) -> list[tuple[int, KillfeedAgentDetection]]:
        """
        Detect killer and victim from a single killfeed entry.

        Returns ALL candidate combinations above threshold for validation,
        tagged with entry index.

        Args:
            entry_crop: Cropped killfeed entry
            entry_idx: Entry index for logging

        Returns:
            List of (entry_idx, KillfeedAgentDetection) tuples sorted by confidence (high to low)
        """
        # Fast blank-entry check: uniform/near-blank crops have no killfeed content.
        # Saves ~280 template matches per frame when the killfeed is empty.
        gray = cv2.cvtColor(entry_crop, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 8.0:
            return []

        # Step 1: Match killer (all candidates above threshold)
        killer_candidates = self._match_all_templates_candidates(entry_crop, flipped=False)

        if not killer_candidates:
            log.debug(f"Entry {entry_idx}: no killer detected")
            return []

        # Step 2: Match victim (all candidates above threshold)
        victim_candidates = self._match_all_templates_candidates(entry_crop, flipped=True)

        if not victim_candidates:
            log.debug(
                f"Entry {entry_idx}: {len(killer_candidates)} killer candidates "
                f"but no victim detected"
            )
            return []

        # Generate all combinations of killer × victim, tagged with entry index
        detections = []
        for killer_agent, killer_side, killer_confidence in killer_candidates:
            for victim_agent, victim_side, victim_confidence in victim_candidates:
                # Use minimum confidence between killer and victim
                confidence = min(killer_confidence, victim_confidence)

                detection = KillfeedAgentDetection(
                    killer_agent=killer_agent,
                    killer_side=killer_side,
                    victim_agent=victim_agent,
                    victim_side=victim_side,
                    confidence=confidence,
                )
                detections.append((entry_idx, detection))

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d[1].confidence, reverse=True)

        log.debug(
            f"Entry {entry_idx}: Generated {len(detections)} candidate combinations "
            f"({len(killer_candidates)} killers × {len(victim_candidates)} victims)"
        )

        return detections

    def _match_all_templates_candidates(
        self, crop: np.ndarray, flipped: bool = False
    ) -> list[tuple[str, str, float]]:
        """
        Match all agent templates and return ALL candidates above threshold.

        Args:
            crop: Cropped killfeed entry
            flipped: If True, flip templates horizontally before matching (for victim)

        Returns:
            List of (agent_name, side, confidence) tuples sorted by confidence (high to low)
        """
        preprocessed = self._preprocess_crop(crop)
        candidates = []

        # Try matching each template
        for template_key, template_data in self.templates.items():
            template_img = template_data["image"]
            agent_name = template_data["agent"]
            side = template_data["side"]

            # Flip template if needed (for victim detection)
            if flipped:
                template_img = cv2.flip(template_img, 1)

            # Skip if template is larger than crop
            if template_img.shape[0] > preprocessed.shape[0] or template_img.shape[1] > preprocessed.shape[1]:
                continue

            # Color template matching
            result = cv2.matchTemplate(preprocessed, template_img, self.match_method)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            confidence = float(max_val)

            # Clamp to [0, 1] range
            confidence = max(0.0, min(1.0, confidence))

            # Add to candidates if above threshold
            if confidence >= self.min_confidence:
                candidates.append((agent_name, side, confidence))
                log.debug(
                    f"Template {template_key} ({'flipped' if flipped else 'normal'}): "
                    f"conf={confidence:.3f} (ABOVE THRESHOLD)"
                )

        # Sort by confidence (highest first)
        candidates.sort(key=lambda c: c[2], reverse=True)

        return candidates

    def _match_all_templates(
        self, crop: np.ndarray, flipped: bool = False
    ) -> Optional[tuple[str, str, float]]:
        """
        Match all agent templates against a crop and return the best match.

        Args:
            crop: Cropped killfeed entry
            flipped: If True, flip templates horizontally before matching (for victim)

        Returns:
            Tuple of (agent_name, side, confidence) for best match, or None if no match
        """
        # No preprocessing - use color templates directly
        preprocessed = self._preprocess_crop(crop)

        best_confidence = 0.0
        best_agent = None
        best_side = None

        # Try matching each template
        for template_key, template_data in self.templates.items():
            template_img = template_data["image"]
            agent_name = template_data["agent"]
            side = template_data["side"]

            # Flip template if needed (for victim detection)
            if flipped:
                template_img = cv2.flip(template_img, 1)

            # Skip if template is larger than crop (can't match)
            if template_img.shape[0] > preprocessed.shape[0] or template_img.shape[1] > preprocessed.shape[1]:
                log.debug(
                    f"Template {template_key} ({template_img.shape[:2]}) larger than crop "
                    f"({preprocessed.shape[:2]}), skipping"
                )
                continue

            # Color template matching - slides template across crop
            result = cv2.matchTemplate(preprocessed, template_img, self.match_method)

            # Get the best match location
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            confidence = float(max_val)

            # Clamp to [0, 1] range
            confidence = max(0.0, min(1.0, confidence))

            log.debug(
                f"Template {template_key} ({'flipped' if flipped else 'normal'}): "
                f"confidence = {confidence:.3f} at position {max_loc}"
            )

            # Track best match
            if confidence > best_confidence:
                best_confidence = confidence
                best_agent = agent_name
                best_side = side

        # Check if best match meets threshold
        if best_confidence < self.min_confidence:
            log.debug(
                f"Best match confidence {best_confidence:.2f} below threshold {self.min_confidence}"
            )
            return None

        return best_agent, best_side, best_confidence

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess killfeed crop for template matching.

        Uses minimal preprocessing to preserve color detail.

        Args:
            crop: Cropped killfeed entry

        Returns:
            Preprocessed color image
        """
        # Ensure BGR color format (OpenCV default)
        if len(crop.shape) == 2:
            # Grayscale - convert to BGR
            preprocessed = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            # RGBA - convert to BGR
            preprocessed = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        else:
            # Already BGR
            preprocessed = crop.copy()

        return preprocessed
