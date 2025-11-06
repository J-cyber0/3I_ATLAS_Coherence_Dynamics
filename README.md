# 3I/ATLAS — Coherence Dynamics (F1–F4)
Minimal, runnable scaffold that generates Figures **F1–F4** discussed in the NGA/coherence analysis.

## What this includes
- **F1**: Polar histogram (rose-like) of NGA azimuth around the ecliptic normal
- **F2**: Rolling NGA components A1, A2, A3 (toy rolling-window fit)
- **F3**: Injection–recovery power curve (toy)
- **F4**: Model comparison ΔBIC bars (toy)

> Note: This is a self-contained demo that doesn't depend on real orbit solvers yet. It is structured so you can drop-in real fitting code later without changing the figure scripts.

## Quick start
```bash
pip install -r requirements.txt
python run_all.py
```
Outputs will appear in `figures/` and `results/`.

## Layout
```
data/
  astrometry.csv         # synthetic astrometry
src/
  orbit_fit.py           # stub fitter returning synthetic NGA & BIC
  orientation.py         # angle stats relative to ecliptic/Jupiter planes
  models.py              # simple ΔBIC comparison helpers
  mc_resample.py         # synthetic re-samples for F1 orientation
  make_figures.py        # generates F1–F4
figures/
results/
run_all.py
requirements.txt
```

## Next steps
- Swap the stubs in `orbit_fit.py` with real orbit+NGA fits.
- Replace synthetic re-samples with astrometric covariance-driven draws.
- Expand F2 to true rolling-window fits; add θ(t) to planes.
- Add physical model grids (radiation pressure, outgassing laws) and replot F4.
