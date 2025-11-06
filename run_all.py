#!/usr/bin/env python3
import os
import json
import pandas as pd
from datetime import datetime
from src.make_figures import make_figures


def main():
    """
    Master entry point to run all analyses for interstellar objects.

    Each object gets its own output directories, seeds, and summary report.
    """
    # --- Environment setup ---
    base_fig = "figures"
    base_res = "results"
    os.makedirs(base_fig, exist_ok=True)
    os.makedirs(base_res, exist_ok=True)

    # --- Object registry (deterministic seed mapping) ---
    objects = {
        "1I": {"path": "data/1I_vectors.csv", "seed": 101},
        "2I": {"path": "data/2I_vectors.csv", "seed": 202},
        "3I": {"path": "data/3I_vectors.csv", "seed": 303},
    }

    all_reports = {}
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n🚀 Starting analysis run — {timestamp}")
    print(f"Output: {base_fig}/ and {base_res}/\n")

    # --- Loop through each object ---
    for key, cfg in objects.items():
        data_path = cfg["path"]
        seed = cfg["seed"]

        if not os.path.exists(data_path):
            print(f"⚠️  Skipping {key}: missing data file → {data_path}")
            continue

        print(f"=== Processing {key} ===")

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            print(f"❌ Could not load {data_path}: {e}")
            continue

        # Per-object subfolders
        fig_dir = os.path.join(base_fig, key)
        res_dir = os.path.join(base_res, key)
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        # --- Run full figure generation ---
        try:
            report = make_figures(
                df_ast=df,
                out_dir=fig_dir,
                results_dir=res_dir,
                seed=seed,
                object_name=key
            )
            all_reports[key] = report
            print(f"✅ {key} complete — results stored in {res_dir}/\n")

        except Exception as e:
            print(f"❌ Error while processing {key}: {e}\n")

    # --- Write combined summary ---
    summary_path = os.path.join(base_res, "summary_all.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "objects_processed": list(all_reports.keys()),
                    "results": all_reports
                },
                f,
                indent=2
            )
        print(f"🧾 All summaries saved to {summary_path}")
    except Exception as e:
        print(f"❌ Failed to write combined summary: {e}")

    if not all_reports:
        print("⚠️  No objects were processed successfully.")


if __name__ == "__main__":
    main()
