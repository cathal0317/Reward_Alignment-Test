"""
Reward utilities for F5-TTS.

We keep imports light so that optional dependencies (e.g. librosa for WavLM-based S-SIM)
do not break ECAPA-based reward usage.
"""

# ECAPA-based S-SIM (no extra deps beyond torch/torchaudio)
from .compute_ssim import (  # noqa: F401
    ECAPASpeakerReward,
    compute_ecapa_ssim_from_paths,
)

# WavLM-based S-SIM (requires librosa and Evaluation_Metric-style speaker backend)
try:  # noqa: SIM105
    from .wavlm_ssim import (  # type: ignore[import]  # noqa: F401
        WavLMSpeakerReward,
        compute_wavlm_ssim_from_arrays,
        compute_wavlm_ssim_from_paths,
    )
except Exception:  # pragma: no cover - optional dependency
    # Allow using ECAPA rewards even if librosa / Evaluation_Metric are not installed.
    WavLMSpeakerReward = None  # type: ignore[assignment]
    compute_wavlm_ssim_from_arrays = None  # type: ignore[assignment]
    compute_wavlm_ssim_from_paths = None  # type: ignore[assignment]

