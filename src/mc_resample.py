import numpy as np
import pandas as pd
import warnings


def resample_and_fit(
    n_samples,
    fit_fn,
    df_astrometry=None,
    seed=0,
    object_name=None,
    sigma_pos=1e-5,      # ↑ slightly larger: avoids underflow
    sigma_vel=1e-7,
    subset_frac=0.7,
    max_fail_frac=0.05,
    tol_duplicate=1e-10, # ↓ looser tolerance for distinguishing near-identical vectors
    adaptive_sigma=True,
    verbose=False,
):
    """
    Bootstrap non-gravitational acceleration fits via random resampling
    and small perturbations of astrometry.

    Parameters
    ----------
    n_samples : int
        Number of bootstrap fits.
    fit_fn : callable
        Function handle, e.g. `fit_orbit_with_nga`.
    df_astrometry : DataFrame
        Columns: JD_TDB, x_au, y_au, z_au, vx_au_d, vy_au_d, vz_au_d.
    seed : int
        Base random seed.
    object_name : str
        "1I", "2I", or "3I" — ensures deterministic independence.
    sigma_pos : float
        Positional perturbation amplitude [au].
    sigma_vel : float
        Velocity perturbation amplitude [au/day].
    subset_frac : float
        Fraction of samples per bootstrap (0–1).
    max_fail_frac : float
        Allowed fraction of fit failures before aborting.
    tol_duplicate : float
        Duplicate-filter tolerance.
    adaptive_sigma : bool
        If True, expands σ after heavy duplication.
    verbose : bool
        Print periodic progress.

    Returns
    -------
    np.ndarray (N,3)
        Array of [A1, A2, A3] fitted acceleration vectors.
    """
    if df_astrometry is None or len(df_astrometry) < 4:
        raise ValueError("Valid df_astrometry must be provided with ≥4 rows.")

    # ------------------------------------------------------------------
    # Deterministic, decorrelated seeding
    # ------------------------------------------------------------------
    SEEDS = {"1I": 101, "2I": 202, "3I": 303}
    base_seed = SEEDS.get(object_name, seed)
    rng = np.random.default_rng(base_seed + 17)  # offset decorrelates objects

    n_rows = len(df_astrometry)
    subset_size = max(3, int(subset_frac * n_rows))

    A_vectors = []
    fail_count = 0

    # ------------------------------------------------------------------
    # Bootstrap loop
    # ------------------------------------------------------------------
    for i in range(n_samples):
        idx = rng.choice(n_rows, size=subset_size, replace=False)
        df_sub = df_astrometry.iloc[np.sort(idx)].copy()

        # Gaussian perturbations
        df_sub[["x_au", "y_au", "z_au"]] += rng.normal(0, sigma_pos, (subset_size, 3))
        df_sub[["vx_au_d", "vy_au_d", "vz_au_d"]] += rng.normal(0, sigma_vel, (subset_size, 3))

        try:
            fit = fit_fn(df_sub, with_nga=True, seed=base_seed + 1000 + i)
            A_vectors.append([fit["A1"], fit["A2"], fit["A3"]])
        except Exception as e:
            fail_count += 1
            if verbose and fail_count <= 5:
                warnings.warn(f"[{object_name}] fit failed at sample {i+1}: {e}")
            continue

        if verbose and (i + 1) % 100 == 0:
            print(f"  {object_name}: {i+1}/{n_samples} resamples complete")

    if not A_vectors:
        raise RuntimeError(f"{object_name}: no successful fits returned.")

    if fail_count / n_samples > max_fail_frac:
        raise RuntimeError(
            f"{object_name}: too many fit failures ({fail_count}/{n_samples})."
        )

    A_array = np.array(A_vectors)

    # ------------------------------------------------------------------
    # Duplicate filtering
    # ------------------------------------------------------------------
    rounded = np.round(A_array / tol_duplicate).astype(int)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    A_unique = A_array[np.sort(unique_idx)]

    if adaptive_sigma and len(A_unique) < 0.9 * len(A_array):
        sigma_pos *= 1.5
        sigma_vel *= 1.5
        if verbose:
            print(
                f"⚠️  {object_name}: {len(A_array) - len(A_unique)} duplicates removed "
                f"({len(A_unique)} unique). Increasing σ_pos, σ_vel → "
                f"{sigma_pos:.1e}, {sigma_vel:.1e}"
            )

    if verbose:
        print(
            f"  {object_name}: {len(A_unique)} unique NGA vectors "
            f"(failures: {fail_count}, duplicates: {len(A_array) - len(A_unique)})"
        )

    # ------------------------------------------------------------------
    # Quick sanity diagnostic
    # ------------------------------------------------------------------
    mean_A = np.mean(A_unique, axis=0)
    if verbose:
        print(f"  {object_name} mean NGA = {mean_A}")

    return A_unique
