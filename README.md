## Reward_Alignment-Test
## 1. Setup

### Create environment and install dependencies

```bash
conda create -n reward-tts python=3.11
conda activate reward-tts

# Install PyTorch with the right CUDA/ROCm/CPU build, e.g. (adjust as needed):
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Install this repo as an editable package
cd /springbrook/share/maths/mausfc/Reward_Alignment-Test
pip install -e .
```


## 2. Reward: ECAPA-based S‑SIM (speaker similarity)

Added ssim evaluation `src/f5_tts/reward/` that reuses the **ECAPA‑TDNN + WavLM speaker verification backend** from the original F5‑TTS eval code to compute **speaker similarity (S‑SIM)** between two wav files.

- `src/f5_tts/reward/compute_ssim.py`:
  - **`ECAPASpeakerReward`**
    - Loads the ECAPA‑TDNN model via `ECAPA_TDNN_SMALL` (from `f5_tts.eval.ecapa_tdnn`) with:
      - `feat_type="wavlm_large"`, `sr=16000`
    - Exposes:
      - `from_paths(gen_wav: str | Path, ref_wav: str | Path) -> float`  
        Returns **speaker similarity (S‑SIM)** as a cosine similarity in \([-1, 1]\).
  - **`compute_ecapa_ssim_from_paths(gen_wav, ref_wav, ckpt_path, device=None) -> float`**  
    One-shot functional API that internally constructs `ECAPASpeakerReward`.
- `src/f5_tts/reward/ssim_test.py`:
  - Simple CLI wrapper to test S‑SIM for a single pair of wav files.


### Downloading the WavLM / ECAPA checkpoint

The ECAPA wrapper expects the same UniSpeech WavLM Large (speaker verification) checkpoint as used by the original F5‑TTS eval scripts.

1. Open the upstream F5‑TTS Evaluation README:  
   [`https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/eval/README.md`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/eval/README.md)
2. Follow the link there to download **`wavlm_large_finetune.pth`** (UniSpeech WavLM-SV checkpoint).
3. Place the file under this repo at:

   ```text
   Reward_Alignment-Test/checkpoints/UniSpeech/wavlm_large_finetune.pth
   ```


### Quick S‑SIM test (CLI)

From the repo root:

```bash

PYTHONPATH=src:$PYTHONPATH python -m f5_tts.reward.ssim_test \
  --ckpt_path checkpoints/UniSpeech/wavlm_large_finetune.pth \
  --ref_wav inference_input/Angry_1.wav \
  --gen_wav inference_input/Sad_female.wav
```

This prints a scalar S‑SIM score between the generated and reference wavs (cosine similarity of ECAPA embeddings).



