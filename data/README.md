# 3I/ATLAS — Horizons Fetch (Heliocentric)

This script fetches REAL state vectors for 3I/ATLAS from NASA JPL Horizons in a heliocentric frame and saves a CSV compatible with your repo.

## What it does
- Queries Horizons for object “C/2025 N1 (ATLAS)” (and fallback identifiers)  
- Center: `@sun` (heliocentric)  
- Table: **VECTORS** (gives X, Y, Z, VX, VY, VZ)  
- Time span: 1900-01-01 to 2100-01-01 by default (you can shorten).  
- Output: `data/astrometry.csv` with columns:  
  `JD_TDB,x_au,y_au,z_au,vx_au_d,vy_au_d,vz_au_d,r_au`

## Install
```
pip install astroquery pandas
```

## Run (inside your project root)
```
python fetch_horizons.py --start 2025-03-01 --stop 2025-12-31 --step 1d --out data/astrometry.csv
```

If the main identifier fails, the script tries: 
- "C/2025 N1 (ATLAS)"
- "C/2025 N1"
- "3I/ATLAS"
