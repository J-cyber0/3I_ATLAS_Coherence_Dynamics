# 3I/ATLAS — Coherence Dynamics (F1–F4)
Open framework for exploring **non-gravitational acceleration (NGA)** and **coherence geometry** across interstellar objects, beginning with **3I/ATLAS**.  
This repository generates Figures **F1–F4**, the first diagnostics from the coherence and NGA analysis pipeline.  

The project builds on a shared MCMC fitting structure used for 1I/ʻOumuamua and 2I/Borisov, aiming to test whether 3I’s persistent **in-plane acceleration** and **near-ecliptic orientation** represent an outlier or part of a larger coherent trend.

---

## Overview
Each figure in this framework highlights a specific aspect of the coherence study:

- **F1** — Polar histogram of NGA azimuth around the ecliptic normal, showing the planar bias.  
- **F2** — Rolling NGA components (*A1, A2, A3*) across perihelion, illustrating the smooth thermal-like onset and persistence.  
- **F3** — Injection–recovery curve to visualize detection sensitivity under synthetic noise.  
- **F4** — ΔBIC comparison of gravity-only, comet-law, and impulsive-jet models, reflecting model selection outcomes from the 3I fit.

While the current version uses synthetic placeholders, the structure is ready to accept real orbit-fit outputs, covariance propagation, and physical model grids.

---

## Scientific Context
Preliminary fits suggest:
- Radial acceleration peaks near **3 × 10⁻⁷ au·d⁻²**, remaining steady through perihelion.  
- Transverse amplitude tracks at ~40–50 % of radial, slightly lagged.  
- Normal component is statistically zero (*σ ≈ 2 × 10⁻⁸ au·d⁻²*), indicating strict in-plane confinement.  
- Orientation aligns to within **0.5°** of both the ecliptic and Jupiter’s Laplace plane — a roughly **1-in-33** probability under isotropy.  

Together, these patterns form a bridge between 1I’s oblique, non-thermal impulse and 2I’s continuous thermal profile — suggesting that 3I occupies a transitional regime of **thermal onset with directional confinement**.

---

## Quick Start
```bash
pip install -r requirements.txt
python run_all.py
```
Figures and results are written to `figures/` and `results/`.

---

## Repository Layout
```
data/
  1I_vectors.csv        # orbital vector data for 1I/ʻOumuamua
  2I_vectors.csv        # orbital vector data for 2I/Borisov
  3I_vectors.csv        # orbital vector data for 3I/ATLAS
src/
  orbit_fit.py           # for orbit+NGA fitting
  orientation.py         # computes plane alignment vs. ecliptic/Laplace planes
  models.py              # ΔBIC comparison and model diagnostics
  mc_resample.py         # isotropic Monte-Carlo resampling for F1
  make_figures.py        # orchestrates F1–F4 generation
fetch_horizons_async.py  # async JPL Horizons client for real orbital/astrometry data
figures/
results/
run_all.py               # master runner for all objects (1I, 2I, 3I)
requirements.txt
```

---

## How `run_all.py` Works
The script **`run_all.py`** is the master entry point for the analysis.  
It loops through all interstellar objects (1I, 2I, and 3I), loads their corresponding datasets from `data/`, and generates diagnostic figures and summary reports.  

**Core behavior:**
- Creates per-object folders in `figures/` and `results/`  
- Calls `make_figures()` for each target with deterministic random seeds  
- Writes per-object reports and a combined summary in `results/summary_all.json`  
- Skips missing or unreadable datasets gracefully  

**Additional command-line arguments:**
```bash
python run_all.py --target 3I            # Run a single object only (e.g., 1I, 2I, all)
python run_all.py --overwrite            # Recompute and overwrite existing results
python run_all.py --quiet                # Suppress figure generation logs
```
If no arguments are provided, the script processes all available datasets in sequence.

Example console output:
```
🚀 Starting analysis run — 2025-11-06 18:00 UTC
Output: figures/ and results/

=== Processing 3I ===
✅ 3I complete — results stored in results/3I/
🧾 All summaries saved to results/summary_all.json
```

---

## Fetching Real Data
The script **`fetch_horizons_async.py`** connects to [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) using asynchronous requests to retrieve orbital elements, ephemerides, and observational data for objects like 1I, 2I, and 3I.  
Outputs are written to `data/`, where `run_all.py` automatically detects and processes them.

Example usage:
```bash
python fetch_horizons_async.py --target 3I --start 2025-01-01 --stop 2025-12-31 --step 1d
```

Optional arguments:
- `--target` — Object name or ID (e.g., 1I, 2I, 3I)  
- `--start`, `--stop` — Observation window (ISO format)  
- `--step` — Step size for Horizons query (e.g., 1d, 6h)  
- `--out` — Output CSV path (default: data/<target>_vectors.csv)  

---

> This framework aims to provide a transparent, reproducible path for studying in-plane acceleration coherence across interstellar objects and refining our understanding of their underlying physical drivers.
