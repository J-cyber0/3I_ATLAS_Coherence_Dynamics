import numpy as np
import pandas as pd


# ============================================================
#  KEPLERIAN ACCELERATION
# ============================================================
def keplerian_accel(x, mu=1.32712440018e11):
    """
    Compute Keplerian acceleration [au/day^2] for one or more position vectors [au].

    Parameters
    ----------
    x : array_like
        3-vector or (N, 3) array of heliocentric positions [au].
    mu : float, optional
        Gravitational parameter of the Sun [km^3/s^2].
        Default = 1.32712440018e11.

    Returns
    -------
    a : ndarray
        Acceleration vector(s) [au/day^2] with same shape as x.
    """
    AU_KM = 1.495978707e8
    SECONDS_PER_DAY = 86400.0
    mu_au_d2 = mu / AU_KM**3 * SECONDS_PER_DAY**2

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        r = np.linalg.norm(x)
        if r < 1e-12:
            return np.zeros(3)
        return -mu_au_d2 * x / r**3

    r = np.linalg.norm(x, axis=1)
    r_safe = np.clip(r, 1e-12, None)
    return -mu_au_d2 * x / r_safe[:, None]**3


# ============================================================
#  BIC CALCULATOR
# ============================================================
def bic(chi2_val, n, k, assume_known_variance=True):
    """
    Bayesian Information Criterion.

    If assume_known_variance=True (recommended for weighted least squares):
        BIC = chi2_val + k*log(n)
    Otherwise (variance estimated from data):
        var_hat = chi2_val / (n - k)
        BIC = n*log(var_hat) + k*log(n)
    """
    if n <= 0 or np.isnan(chi2_val):
        return np.nan
    if assume_known_variance:
        return float(chi2_val + k * np.log(n))
    var_hat = chi2_val / max(n - k, 1)
    return float(n * np.log(var_hat) + k * np.log(n))


# ============================================================
#  INTERNAL HELPER FOR WEIGHTED LINEAR FIT
# ============================================================
def _fit_linear_model(X, y, w):
    """
    Weighted least-squares fit returning coef, Aerr, chi2_w, var_hat, bic.

    Parameters
    ----------
    X : ndarray
        Design matrix (N, 3)
    y : ndarray
        Flattened acceleration residuals (N,)
    w : ndarray
        Weights (N,)

    Returns
    -------
    coef, Aerr, chi2_w, var_hat, bic_val
    """
    Xw = X * np.sqrt(w[:, None])
    yw = y * np.sqrt(w)

    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ coef
    chi2_w = float(np.sum(resid**2))
    n_obs, k = len(yw), X.shape[1]

    var_hat = chi2_w / max(n_obs - k, 1)
    bic_val = n_obs * np.log(var_hat) + k * np.log(n_obs)

    cov = np.linalg.pinv(Xw.T @ Xw) * var_hat
    Aerr = np.sqrt(np.diag(cov))
    return coef, Aerr, chi2_w, var_hat, bic_val


# ============================================================
#  ORBIT FITTING WITH OPTIONAL NON-GRAVITATIONAL ACCELERATION
# ============================================================
def fit_orbit_with_nga(df_astrometry, with_nga=True, seed=0):
    """
    Fit constant non-gravitational acceleration components (A1, A2, A3)
    using weighted least squares on interval accelerations.

    Robust to irregular JD steps and usable for sensitivity analysis.
    """
    n_all = len(df_astrometry)
    if n_all < 4:
        return {"A1": 0.0, "A2": 0.0, "A3": 0.0,
                "Aerr": [np.nan]*3, "chi2": np.nan,
                "bic": np.nan, "rms_resid": np.nan}

    jd = df_astrometry["JD_TDB"].to_numpy()
    pos = df_astrometry[["x_au", "y_au", "z_au"]].to_numpy()
    vel = df_astrometry[["vx_au_d", "vy_au_d", "vz_au_d"]].to_numpy()

    dt_all = np.diff(jd)
    mask = dt_all > 0
    if not np.any(mask):
        return {"A1": 0.0, "A2": 0.0, "A3": 0.0,
                "Aerr": [np.nan]*3, "chi2": np.nan,
                "bic": np.nan, "rms_resid": np.nan}

    vel_0 = vel[:-1][mask]
    vel_1 = vel[1:][mask]
    dt = np.clip(dt_all[mask], 1e-8, None)
    pos_mid = 0.5 * (pos[1:][mask] + pos[:-1][mask])

    a_obs = (vel_1 - vel_0) / dt[:, None]
    r_norm = np.clip(np.linalg.norm(pos_mid, axis=1), 1e-12, None)
    a_kep = -0.0002959122082855911 * pos_mid / (r_norm[:, None] ** 3)
    a_resid = a_obs - a_kep

    n_eff = len(a_resid)
    if n_eff < 3:
        return {"A1": 0.0, "A2": 0.0, "A3": 0.0,
                "Aerr": [np.nan]*3, "chi2": np.nan,
                "bic": np.nan, "rms_resid": np.nan}

    n_obs = 3 * n_eff
    w = 1.0 / (dt ** 2)
    W_big = np.repeat(w, 3)
    W_big *= (3 * n_eff) / np.sum(W_big)
    W_big = np.minimum(W_big, np.median(W_big) * 25)

    # ============ GRAVITY-ONLY MODEL ============
    if not with_nga:
        y = a_resid.reshape(-1)
        _, _, chi2_w0, var_hat0, bic_val0 = _fit_linear_model(
            np.zeros((len(y), 3)), y, W_big
        )
        return {
            "A1": 0.0, "A2": 0.0, "A3": 0.0,
            "Aerr": [np.nan]*3,
            "chi2": chi2_w0,
            "bic": bic_val0,
            "rms_resid": np.sqrt(var_hat0),
        }

    # ============ NGA MODEL ============
    X = np.tile(np.eye(3), (n_eff, 1))
    y = a_resid.reshape(-1)
    coef, Aerr, chi2_w, var_hat, bic_val = _fit_linear_model(X, y, W_big)

    # Randomized seed scaling for reproducibility in bootstraps
    rng = np.random.default_rng(seed)
    A_vec = coef * (1 + rng.normal(0, 0.01, size=3))
    A1, A2, A3 = map(float, A_vec.ravel()[:3])

    if seed == 0:
        mean_r = float(np.mean(r_norm))
        print(f"[fit_orbit_with_nga] ⟨r⟩={mean_r:.3f} au → "
              f"A=({A1:.3e}, {A2:.3e}, {A3:.3e}); "
              f"χ²={chi2_w:.3e}, σ²={var_hat:.3e}, BIC={bic_val:.2f}")

    return {
        "A1": A1,
        "A2": A2,
        "A3": A3,
        "Aerr": [float(x) for x in Aerr],
        "chi2": float(chi2_w),
        "bic": float(bic_val),
        "rms_resid": float(np.sqrt(var_hat)),
        "r_norm": r_norm.tolist(),
    }


# ============================================================
#  FUTURE HOOKS
# ============================================================
# - Apply rolling JD window (window_sensitivity)
# - Refit leaving out each epoch (leave_one_out)
# - Compare comet-law vs impulsive-jet in models.compare_models
