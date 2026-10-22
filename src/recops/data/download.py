"""Download Yambda parquet files from the Hugging Face Hub (needs internet)."""

from __future__ import annotations

from pathlib import Path

from recops.config import Config

FILES = ("listens", "likes")


def download_yambda(cfg: Config) -> dict[str, Path]:
    """Fetch ``listens.parquet`` and ``likes.parquet`` for the configured Yambda size."""
    from huggingface_hub import hf_hub_download

    out: dict[str, Path] = {}
    for name in FILES:
        path = hf_hub_download(
            repo_id=cfg.data.dataset_repo,
            repo_type="dataset",
            filename=f"{cfg.data.subdir}/{name}.parquet",
            local_dir=str(cfg.paths.data_raw),
        )
        out[name] = Path(path)
        print(f"downloaded {name}: {path}")
    return out
