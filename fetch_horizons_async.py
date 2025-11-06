#!/usr/bin/env python3
"""
Asynchronously fetch heliocentric state vectors from JPL Horizons for
interstellar objects 1I/‘Oumuamua, 2I/Borisov, and 3I/ATLAS.

Each object’s data are written as CSV under the specified output directory.
"""

import argparse
import sys
import os
import asyncio
import random
import pandas as pd
from datetime import datetime


# ----------------------------------------------------------
# Blocking function for single Horizons query
# ----------------------------------------------------------
def fetch_vectors(identifier, start, stop, step, center="@sun", id_type="designation"):
    """Fetch heliocentric state vectors (with acceleration) via astroquery.JPLHorizons."""
    from astroquery.jplhorizons import Horizons
    import numpy as np

    obj = Horizons(id=identifier, id_type=id_type, location=center,
                   epochs={"start": start, "stop": stop, "step": step})
    vec = obj.vectors()

    jd = vec["datetime_jd"]
    x, y, z = vec["x"], vec["y"], vec["z"]
    vx, vy, vz = vec["vx"], vec["vy"], vec["vz"]
    r = vec["r"] if "r" in vec.colnames else np.sqrt(x**2 + y**2 + z**2)

    # ✅ optional accelerations (only present for SPK-based vectors)
    ax = vec["ax"] if "ax" in vec.colnames else np.zeros_like(x)
    ay = vec["ay"] if "ay" in vec.colnames else np.zeros_like(y)
    az = vec["az"] if "az" in vec.colnames else np.zeros_like(z)

    return pd.DataFrame({
        "JD_TDB": jd,
        "x_au": x,
        "y_au": y,
        "z_au": z,
        "vx_au_d": vx,
        "vy_au_d": vy,
        "vz_au_d": vz,
        "ax_au_d2": ax,
        "ay_au_d2": ay,
        "az_au_d2": az,
        "r_au": r
    })

# ----------------------------------------------------------
# Async wrapper with throttling and retries
# ----------------------------------------------------------
async def fetch_one_async(key, identifiers, id_type, start, stop, step, outdir, sem, retries=3):
    async with sem:
        for ident in identifiers:
            for attempt in range(1, retries + 1):
                try:
                    await asyncio.sleep(random.uniform(0.4, 1.2))  # avoid API rate limits
                    df = await asyncio.to_thread(fetch_vectors, ident, start, stop, step, id_type=id_type)

                    jd_min, jd_max = float(df["JD_TDB"].min()), float(df["JD_TDB"].max())
                    if jd_max <= jd_min:
                        raise ValueError(f"Non-increasing JD range for {ident}")

                    out_path = os.path.join(outdir, f"{key}_vectors.csv")
                    if os.path.exists(out_path):
                        old = pd.read_csv(out_path)
                        if df.equals(old):
                            print(f"⚠️  Duplicate data detected for {key}; skipping write.")
                            return True

                    df.to_csv(out_path, index=False)
                    print(f"✅ [{key}] {ident} → {len(df)} rows written")
                    print(f"   JD range: {jd_min:.2f} – {jd_max:.2f}\n")
                    return True

                except Exception as e:
                    print(f"❌ [{key}] attempt {attempt}/{retries} failed for {ident}: {e}", file=sys.stderr)
                    if attempt < retries:
                        await asyncio.sleep(1.5 * attempt)
                        continue
            print(f"⛔ [{key}] all retries failed for {ident}", file=sys.stderr)
        return False


# ----------------------------------------------------------
# Async main loop
# ----------------------------------------------------------
async def main_async(args):
    os.makedirs(args.outdir, exist_ok=True)

    targets = {
        "1I": (["1I/2017 U1", "A/2017 U1", "1I"], "smallbody"),
        "2I": (["2I/Borisov", "C/2019 Q4 (Borisov)", "2I"], "smallbody"),
        "3I": (["C/2025 N1 (ATLAS)", "3I/ATLAS", "C/2025 N1"], "smallbody"),
    }


    selected = targets.keys() if args.object == "all" else [args.object]
    sem = asyncio.Semaphore(args.max_concurrent)

    print(f"\n🚀 Fetching Horizons vectors ({len(selected)} object(s))")
    print(f"   Output directory: {args.outdir}")
    print(f"   Time range: {args.start} → {args.stop} ({args.step})")
    print("----------------------------------------------------------\n")

    tasks = [
        fetch_one_async(
            key, targets[key][0], targets[key][1],
            args.start, args.stop, args.step, args.outdir, sem
        )
        for key in selected
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = sum(bool(r) for r in results if not isinstance(r, Exception))

    print(f"\n🧾 Completed {success_count}/{len(selected)} successful fetches.")
    print(f"Run finished at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")


# ----------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Asynchronously fetch heliocentric Horizons vectors for interstellar objects."
    )
    parser.add_argument("--start", default="1900-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--stop", default="2100-01-01", help="Stop date (YYYY-MM-DD)")
    parser.add_argument("--step", default="1d", help="Sampling step, e.g. '1d' or '6h'")
    parser.add_argument("--outdir", default="data", help="Directory for CSV outputs")
    parser.add_argument("--object", choices=["1I", "2I", "3I", "all"], default="all",
                        help="Select specific object or 'all'")
    parser.add_argument("--max-concurrent", type=int, default=2,
                        help="Maximum concurrent fetches (default 2)")
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n⚠️  Cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
