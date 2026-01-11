from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio

from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL


class ECAPASpeakerReward:

    # f5_tts/eval/ecapa_tdnn.py의 evaluation method를 가져옴
    def __init__(self, ckpt_path: str | Path, device: Optional[str] = None):
        ckpt_path = Path("/springbrook/share/maths/mausfc/F5-TTS_Reward-Alignment/checkpoints/UniSpeech/wavlm_large_finetune.pth")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"ECAPA checkpoint not found: {ckpt_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Match eval setup: feat_dim=1024, feat_type="wavlm_large", sr=16000
        self.model = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type="wavlm_large",
            sr=16000,
            config_path=None,
        )
        
        state_dict = torch.load(
            ckpt_path,
            weights_only=True,
            map_location="cpu",
        )
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _embed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        # Resample to 16k as in `run_sim`
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            resample = resample.to(self.device)
            wav = resample(wav.to(self.device))
        else:
            wav = wav.to(self.device)

        emb = self.model(wav)
        # emb: [1, emb_dim]
        return emb

    @torch.no_grad()
    def from_paths(self, gen_wav: str | Path, ref_wav: str | Path) -> float:
   
        gen_wav = Path(gen_wav)
        ref_wav = Path(ref_wav)

        if not gen_wav.is_file():
            raise FileNotFoundError(f"gen_wav not found: {gen_wav}")
        if not ref_wav.is_file():
            raise FileNotFoundError(f"ref_wav not found: {ref_wav}")

        wav1, sr1 = torchaudio.load(str(gen_wav))
        wav2, sr2 = torchaudio.load(str(ref_wav))

        emb1 = self._embed(wav1, sr=sr1)
        emb2 = self._embed(wav2, sr=sr2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        return float(sim)


@torch.no_grad()
def compute_ecapa_ssim_from_paths(
    gen_wav: str | Path,
    ref_wav: str | Path,
    ckpt_path: str | Path,
    device: Optional[str] = None,
) -> float:
 
    reward = ECAPASpeakerReward(ckpt_path=ckpt_path, device=device)
    return reward.from_paths(gen_wav=gen_wav, ref_wav=ref_wav)
