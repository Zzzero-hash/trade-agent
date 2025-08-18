"""Quick verification for schema extraction & metadata enrichment.

Steps:
1. Extract schema from data/features.parquet (or provided path)
2. Train a tiny Ridge model (if config supplied) with data_path injected
3. Print resulting metadata file path & show hashes

Usage (repo root):
python scripts/verify_schema_extraction.py \
    --data data/features.parquet \
    --config configs/ridge_config.json
"""
from __future__ import annotations

import argparse
import glob
import json

from src.data.schema import extract_schema, save_schema
from src.sl.train import train_model_from_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/features.parquet")
    p.add_argument("--config", default="configs/ridge_config.json")
    p.add_argument("--schema-json", default="data/features_schema.json")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[1] Extracting schema from {args.data}")
    schema_res = extract_schema(args.data, sample_rows=1000)
    save_schema(schema_res, args.schema_json)
    line = (
        f"    schema_hash={schema_res.schema_hash}\n"
        f"    data_hash={schema_res.data_hash}\n"
        f"    n_rows={schema_res.n_rows} n_cols={schema_res.n_cols}"
    )
    print(line)

    print("[2] Training model to produce enriched metadata")
    # Inject data_path by temporarily wrapping provided config
    # Note: train_model_from_config does not pass data_path into SLConfig yet.
    # Hash fields may remain None until SLConfig supports data_path.
    # TODO: Extend SLConfig + pipeline initialization to accept data_path.
    results = train_model_from_config(
        args.config, args.data, target_column="mu_hat"
    )
    print(f"    training metrics keys: {list(results.keys())}")

    # Look for latest metadata file
    meta_files = sorted(glob.glob("models/sl_model_*_metadata.json"))
    if not meta_files:
        print("[WARN] No metadata files found.")
        return
    latest = meta_files[-1]
    with open(latest) as f:
        meta = json.load(f)
    print(f"[3] Loaded metadata {latest}")
    for k in ["schema_hash", "data_hash", "data_n_rows", "data_n_cols"]:
        print(f"    {k}: {meta.get(k)}")
    if meta.get("schema_hash") and meta.get("data_hash"):
        print("[SUCCESS] Metadata contains schema & data hashes")
    else:
        print("[INFO] Hashes missing (no data_path support yet).")


if __name__ == "__main__":
    main()
