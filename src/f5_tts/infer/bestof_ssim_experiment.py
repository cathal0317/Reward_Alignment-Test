from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from importlib.resources import files
from omegaconf import OmegaConf
import tempfile

from f5_tts.infer.utils_infer import (
    device,
    load_model,
    load_vocoder,
    mel_spec_type,
    preprocess_ref_audio_text,
    target_rms,
    target_sample_rate,
    infer_process,
    hop_length,
)
from f5_tts.reward.compute_ssim import ECAPASpeakerReward


MAIN_CONFIG = {
    # Same defaults as midstep_ssim_experiment.py; edit here if needed
    # Use the packaged example ref audio (same as infer_cli default)
    "ref_audio": "/springbrook/share/maths/mausfc/Reward_Alignment-Test/inference_input/Angry_female.wav",
    "ref_text": "Dogs are sitting by the door.",
    "gen_text": (
        "The time varying concentrations of pollutant can be modelled by diffusion-advection reaction equations."
    ),
    # "output_dir": "inference_output_bestof",
    "model_name": "F5TTS_v1_Base",
    "ckpt_step": 1250000,
    "ckpt_type": "safetensors",
    "num_runs": 100,  # number of independent generations (different seeds)
}


def _prepare_model_and_vocoder(
    model_name: str,
    ckpt_step: int,
    ckpt_type: str,
) -> Tuple[torch.nn.Module, object]:
    """Load F5-TTS CFM model and vocoder (same style as infer_cli)."""

    # Model config
    model_cfg_path = files("f5_tts").joinpath(f"configs/{model_name}.yaml")
    model_cfg = OmegaConf.load(str(model_cfg_path))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    # Checkpoint (download from HF if needed)
    repo_name = "F5-TTS"
    ckpt_path = str(
        cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}")
    )

    print(f"[INFO] Using model: {model_name}")
    print(f"[INFO] Checkpoint: {ckpt_path}")

    ema_model = load_model(
        model_cls,
        model_arc,
        ckpt_path,
        mel_spec_type=mel_spec_type,
        vocab_file="",
        device=device,
    )

    # Vocoder (download from HF by default)
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    else:
        raise ValueError(f"Unsupported mel_spec_type: {mel_spec_type}")

    vocoder = load_vocoder(
        vocoder_name=mel_spec_type,
        is_local=False,
        local_path=vocoder_local_path,
        device=device,
    )

    return ema_model, vocoder


def _compute_ref_rms(ref_audio_path: Path) -> float:
    audio, sr = torchaudio.load(str(ref_audio_path))
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    return float(rms.item())


def run_bestof_experiment() -> None:
    cfg = MAIN_CONFIG

    repo_root = Path(__file__).resolve().parents[3]  
    ref_audio_path = repo_root / cfg["ref_audio"]
    # Use a temporary directory for intermediate wavs (no long-term storage)
    tmp_root = Path(tempfile.gettempdir()) / "f5_bestof_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    # 1) Load model + vocoder
    ema_model, vocoder = _prepare_model_and_vocoder(
        model_name=cfg["model_name"],
        ckpt_step=cfg["ckpt_step"],
        ckpt_type=cfg["ckpt_type"],
    )

    # 2) Preprocess reference audio/text (same as infer_cli)
    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
        str(ref_audio_path),
        cfg["ref_text"],
        show_info=print,
    )

    # # 3) Compute reference RMS (for logging only)
    # ref_rms = _compute_ref_rms(Path(ref_audio_proc))
    # print(f"[INFO] Reference RMS: {ref_rms:.6f} (target_rms={target_rms})")

    # 4) Prepare reward (ECAPA S-SIM)
    ssim_ckpt = repo_root / "checkpoints/UniSpeech/wavlm_large_finetune.pth"
    print(f"[INFO] ECAPA/WavLM checkpoint: {ssim_ckpt} (exists={ssim_ckpt.is_file()})")
    reward = ECAPASpeakerReward(ckpt_path=ssim_ckpt, device=None)

    print(f"[INFO] Ref_text: {ref_text_proc}")
    print(f"[INFO] Gen_text: {cfg['gen_text']}")

    num_runs: int = cfg["num_runs"]
    sims: List[float] = []

    # 5) Run multiple independent generations
    for run_idx in range(num_runs):
        seed = 1234 + run_idx
        print(f"\n[RUN {run_idx}] seed={seed}")

        # Control randomness for this run
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Standard infer pipeline (same as infer_cli → infer_process)
        audio_segment, sr, _ = infer_process(
            ref_audio=ref_audio_proc,
            ref_text=ref_text_proc,
            gen_text=cfg["gen_text"],
            model_obj=ema_model,
            vocoder=vocoder,
            mel_spec_type=mel_spec_type,
        )

        # Compute S-SIM between generated wav and (processed) reference audio
        # Write to a temporary wav in tmp_root just for ECAPA scoring
        tmp_path = tmp_root / f"tmp_run{run_idx}.wav"
        sf.write(str(tmp_path), audio_segment.astype(np.float32), sr)
        sim = reward.from_paths(gen_wav=tmp_path, ref_wav=ref_audio_proc)
        sims.append(sim)
        print(f"[RUN {run_idx}] S-SIM={sim:.6f}  -> {tmp_path.name}")

    # 6) Summary statistics
    sims_np = np.array(sims, dtype=np.float32)
    print("\n========== Best-of-N S-SIM Summary ==========")
    print(f"N runs          : {num_runs}")
    print(f"S-SIM values    : {sims_np}")
    print(f"Mean S-SIM      : {sims_np.mean():.6f}")
    print(f"Std  S-SIM      : {sims_np.std():.6f}")
    print(f"Max  S-SIM      : {sims_np.max():.6f}")
    print(f"Argmax run idx  : {int(sims_np.argmax())}")
    print("=============================================\n")


def main() -> None:
    run_bestof_experiment()


if __name__ == "__main__":
    main()


