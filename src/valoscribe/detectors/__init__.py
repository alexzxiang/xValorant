"""Detectors for Valorant HUD elements."""

from valoscribe.detectors.cropper import Cropper
from valoscribe.detectors.template_credits_detector import TemplateCreditsDetector
from valoscribe.detectors.preround_credits_detector import PreroundCreditsDetector
from valoscribe.detectors.ultimate_detector import UltimateDetector
from valoscribe.detectors.preround_ability_detector import PreroundAbilityDetector
from valoscribe.detectors.preround_ultimate_detector import PreroundUltimateDetector

__all__ = [
    "Cropper",
    "TemplateCreditsDetector",
    "PreroundCreditsDetector",
    "UltimateDetector",
    "PreroundAbilityDetector",
    "PreroundUltimateDetector",
]
