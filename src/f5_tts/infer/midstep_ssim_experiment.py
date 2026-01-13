from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from importlib.resources import files
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    device,
    hop_length,
    load_model,
    load_vocoder,
    mel_spec_type,
    mel_to_wave,
    preprocess_ref_audio_text,
    target_rms,
    target_sample_rate,
)
from f5_tts.reward.compute_ssim import ECAPASpeakerReward


MAIN_CONFIG = {
    "ref_audio": "src/f5_tts/infer/examples/basic/basic_ref_en.wav",
    "ref_text": "Some calls me nature others call me mother nature.",
    "gen_text": (
        "The time varying concentrations of pollutant can be modelled by "
        "diffusion-advection reaction equations."
    ),
    "output_dir": "inference_output2/midsteps",
    "model_name": "F5TTS_v1_Base",
    "ckpt_step": 1250000,
    "ckpt_type": "safetensors",
    "num_runs": 1,
    "ode_steps": 32,
}


def _prepare_model_and_vocoder(
    model_name: str,
    ckpt_step: int,
    ckpt_type: str,
) -> Tuple[torch.nn.Module, object]:
    """Load F5-TTS CFM model and vocoder similar to infer_cli."""
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


def run_midstep_experiment() -> None:
    cfg = MAIN_CONFIG

    repo_root = Path(__file__).resolve().parents[3]  # .../Reward_Alignment-Test
    ref_audio_path = repo_root / cfg["ref_audio"]
    output_root = repo_root / cfg["output_dir"]
    output_root.mkdir(parents=True, exist_ok=True)

    # 1) Load model + vocoder
    ema_model, vocoder = _prepare_model_and_vocoder(
        model_name=cfg["model_name"],
        ckpt_step=cfg["ckpt_step"],
        ckpt_type=cfg["ckpt_type"],
    )

    # 2) Preprocess reference audio/text (uses ASR if ref_text == "")
    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
        str(ref_audio_path),
        cfg["ref_text"],
        show_info=print,
    )

    # 3) Compute reference RMS (for loudness matching)
    ref_rms = _compute_ref_rms(Path(ref_audio_proc))
    print(f"[INFO] Reference RMS: {ref_rms:.6f} (target_rms={target_rms})")

    # 4) Prepare reward (ECAPA S-SIM)
    ssim_ckpt = repo_root / "checkpoints/UniSpeech/wavlm_large_finetune.pth"
    print(f"[INFO] ECAPA/WavLM checkpoint: {ssim_ckpt} (exists={ssim_ckpt.is_file()})")
    reward = ECAPASpeakerReward(ckpt_path=ssim_ckpt, device=None)

    # 5) Load preprocessed ref_audio as waveform for duration computation
    audio, sr = torchaudio.load(ref_audio_proc)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    ref_audio_len = audio.shape[-1] // hop_length

    print(f"[INFO] Using ref_text: {ref_text_proc}")
    print(f"[INFO] Using gen_text: {cfg['gen_text']}")

    # 6) Run multiple trajectories with different seeds
    nfe_step: int = cfg["ode_steps"]
    num_runs: int = cfg["num_runs"]

    for run_idx in range(num_runs):
        seed = 1234 + run_idx
        print(f"\n[RUN {run_idx}] seed={seed}")

        # Duration estimation (same logic as infer_batch_process)
        ref_text_len = len(ref_text_proc.encode("utf-8"))
        gen_text_len = len(cfg["gen_text"].encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)

        # Prepare inputs for CFM.sample
        ema_model.eval()
        with torch.no_grad():
            cond = ema_model.mel_spec(audio.to(device))  # [1, n_mels, T]
            cond = cond.permute(0, 2, 1)  # [1, T, n_mels]

            out, trajectory = ema_model.sample(
                cond=cond,
                text=[ref_text_proc + cfg["gen_text"]],
                duration=duration,
                steps=nfe_step,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                seed=seed,
                max_duration=4096,
                vocoder=None,
                use_epss=True,
                no_ref_audio=False,
            )

        # trajectory: [steps+1, B, T, C]
        num_traj_steps = trajectory.shape[0] - 1
        print(f"[RUN {run_idx}] trajectory steps: {num_traj_steps}")

        # 7) For all time indices, decode gen-part wav + compute S-SIM
        sims: List[Tuple[int, float]] = []
        for t_idx in range(num_traj_steps + 1):
            xt = trajectory[t_idx][0]  # [T_total, C]

            if xt.shape[0] <= ref_audio_len:
                continue
            xt = trajectory[t_idx][0]  # [T, C]
            gen_t = xt[ref_audio_len:, :]      # Exclude the Reference Audio
            if gen_t.shape[0] == 0:
                continue                      

            mel_t = gen_t.permute(1, 0)

            wav_np = mel_to_wave(
                mel_t,
                vocoder,
                mel_spec_type=mel_spec_type,
                rms=ref_rms,
                target_rms_value=target_rms,
            )

            t_frac = float(t_idx) / float(max(num_traj_steps, 1))
            out_path = output_root / f"run{run_idx}_t{t_idx:02d}_t{t_frac:.2f}.wav"
            sf.write(str(out_path), wav_np.astype(np.float32), target_sample_rate)

            sim = reward.from_paths(gen_wav=out_path, ref_wav=ref_audio_path)
            sims.append((t_idx, sim))
            print(f"[RUN {run_idx}] t_idx={t_idx:02d} (t≈{t_frac:.2f})  S-SIM={sim:.6f}  -> {out_path.name}")

        # 8) Save S-SIM vs timestep and plot
        if sims:
            t_arr = np.array([s[0] for s in sims], dtype=np.float32)
            sim_arr = np.array([s[1] for s in sims], dtype=np.float32)
            np.save(output_root / f"run{run_idx}_ssim_vs_t.npy", np.stack([t_arr, sim_arr], axis=1))

            plt.figure(figsize=(6, 4))
            plt.plot(t_arr, sim_arr, marker="o")
            plt.xlabel("ODE step index (t_idx)")
            plt.ylabel("S-SIM (ECAPA)")
            plt.title(f"S-SIM vs timestep)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_root / f"sim_vs_timestep.png", dpi=150)
            plt.close()


def main() -> None:
    run_midstep_experiment()


if __name__ == "__main__":
    main()


