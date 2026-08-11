"""Script chấm điểm — dùng cho BGK.

    python predict.py <input.csv> <output.csv> [text_column]
"""
import sys, time
from pathlib import Path

import pandas as pd
import torch

from prompt_firewall import PromptFirewall

MODEL_DIR = Path(__file__).parent / "firewall_model"


def main(input_csv, output_csv, text_col="prompt"):
    df = pd.read_csv(input_csv)
    # Chọn thiết bị nhanh nhất hiện có. Bỏ nhánh MPS thì máy Apple Silicon
    # rơi về CPU và chậm ~3x so với số đo trong notebook.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    fw = PromptFirewall.load(MODEL_DIR, device=device)

    prompts = df[text_col].astype(str).tolist()
    fw.predict(prompts[:200])                      # warm-up, không tính giờ

    t0 = time.perf_counter()
    preds = fw.predict(prompts)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    df["label"] = preds                            # 0 = safe, 1 = unsafe
    df.to_csv(output_csv, index=False)
    print(f"{len(df):,} prompt trong {elapsed:.2f}s "
          f"({len(df) / elapsed:,.0f} prompt/s, device={device})")


if __name__ == "__main__":
    main(*sys.argv[1:])
