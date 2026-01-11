from __future__ import annotations

"""
Compute S-SIM between two wav files.

    PYTHONPATH=src:$PYTHONPATH python -m f5_tts.reward.ssim_test \
  --ckpt_path checkpoints/UniSpeech/wavlm_large_finetune.pth \
  --ref_wav inference_input/0020_000337_neutral.wav \
  --gen_wav inference_input/0020_000337_neutral.wav

Used the same backend as f5-tts from
    - f5_tts.eval.ecapa_tdnn.ECAPA_TDNN_SMALL
    - f5_tts.reward.compute_ssim.ECAPASpeakerReward
"""

import argparse
from pathlib import Path

from f5_tts.reward.compute_ssim import ECAPASpeakerReward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ECAPA-based speaker similarity (S-SIM) between two wav files."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to ECAPA/WavLM checkpoint (e.g. checkpoints/UniSpeech/wavlm_large_finetune.pth).",
    )
    parser.add_argument(
        "--ref_wav",
        type=str,
        required=True,
        help="Reference wav path.",
    )
    parser.add_argument(
        "--gen_wav",
        type=str,
        required=True,
        help="Generated wav path to compare against the reference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. cuda:0 or cpu). If omitted, auto-select.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt_path)
    ref_wav = Path(args.ref_wav)
    gen_wav = Path(args.gen_wav)

    print(f"[INFO] ckpt_path: {ckpt_path} (exists={ckpt_path.is_file()})")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    if not ref_wav.is_file():
        raise FileNotFoundError(f"Reference wav not found at: {ref_wav}")
    if not gen_wav.is_file():
        raise FileNotFoundError(f"Generated wav not found at: {gen_wav}")

    reward = ECAPASpeakerReward(ckpt_path=ckpt_path, device=args.device)

    sim = reward.from_paths(gen_wav=gen_wav, ref_wav=ref_wav)
    print(f"S-SIM (ECAPA) between\n  gen: {gen_wav}\n  ref: {ref_wav}\n=> {sim:.6f}")


if __name__ == "__main__":
    main()


