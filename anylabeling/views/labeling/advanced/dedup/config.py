"""Thresholds for near-duplicate cleanup.

Scene change is three separate limits (translation, scale, rotation).
They are not added together, because those quantities have different units.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    thumb_max_side: int = 256
    verify_max_side: int = 1024
    hash_bits: int = 64
    phash_hamming_max: int = 6
    dhash_hamming_max: int = 6
    hash_require_both: bool = True
    box_grid: int = 3
    # Checked labels in the dedup dialog. Used only when instance gate is on.
    box_label_filter: tuple[str, ...] = ()
    box_multilabel: bool = True
    box_max_per_image: int = 64
    box_classes: tuple[str, ...] = ()


@dataclass
class RecallConfig:
    lsh_bands: int = 4
    box_topk: int = 8
    box_sim_min: float = 0.97
    # Exact cosine is used below this size; above it, signature LSH is used.
    box_exact_max: int = 2000
    max_candidates_per_image: int = 128


@dataclass
class VerifyConfig:
    sift_nfeatures: int = 2048
    ratio_test: float = 0.72
    min_inliers: int = 20
    min_inlier_ratio: float = 0.45
    # Tighter than 0.12 so a small overlapping patch is not "the same scene".
    min_overlap: float = 0.35
    max_homography_distortion: float = 0.6
    mask_boxes_when_fitting_h: bool = True
    min_good_after_mask: int = 20


@dataclass
class QualityMetric:
    key: str
    enabled: bool
    rel_tol: float
    abs_tol: float


@dataclass
class QualityConfig:
    enabled: bool = True
    eps: float = 1e-6
    metrics: list[QualityMetric] = field(default_factory=lambda: [
        QualityMetric("sharpness_laplacian_var_log1p", True, 0.20, 0.35),
        QualityMetric("brightness_l_mean", True, 0.20, 2.0),
        QualityMetric("contrast_rms_gray", True, 0.20, 2.0),
        QualityMetric("saturation_mean", True, 0.25, 2.0),
        QualityMetric("value_mean", True, 0.20, 2.0),
        QualityMetric("noise_mad_residual", True, 0.20, 3.0),
        QualityMetric("artifact_heuristic_score", False, 0.20, 5.0),
        QualityMetric("brisque", False, 0.20, 5.0),
    ])


@dataclass
class Gate2Config:
    """Each limit is independent. Exceeding any one keeps the pair as diversity."""

    max_translation: float = 0.02
    max_scale: float = 0.02
    max_rotation_rad: float = 0.03


@dataclass
class InstanceConfig:
    require_count_equal: bool = True
    min_box_score: float = 0.0
    max_true_shift: float = 0.03
    max_scale_diff: float = 0.20
    max_pose_diff: float = 0.20
    use_appearance: bool = False
    min_sift_good: int = 6
    min_hist_sim: float = 0.75
    appearance_is_hard: bool = False


@dataclass
class StrictConfig:
    enabled: bool = True
    gate2: Gate2Config = field(default_factory=Gate2Config)
    instance: InstanceConfig = field(default_factory=InstanceConfig)
    require_quality: bool = True
    # When False, pairs that pass scene motion go straight to the quality gate.
    use_instance_gate: bool = True


@dataclass
class Config:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    strict: StrictConfig = field(default_factory=StrictConfig)