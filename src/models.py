import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ============================================================
#  MODEL COMPARISON (BIC-BASED)
# ============================================================
def compare_models(bic_null, bic_alt_dict, threshold_strong=10.0):
    """
    Compare Bayesian Information Criterion (BIC) values between a null model
    (gravity-only) and one or more alternatives (e.g., NGA, jet, comet-law).

    Parameters
    ----------
    bic_null : float
        BIC for the baseline (gravity-only) model.
    bic_alt_dict : dict
        {model_name: bic_value}
    threshold_strong : float
        ΔBIC threshold where evidence becomes "decisive".

    Returns
    -------
    pd.DataFrame
        model | bic | delta_bic | weight | evidence
    """
    models = {"gravity_only": bic_null}
    models.update(bic_alt_dict)

    df = pd.DataFrame(list(models.items()), columns=["model", "bic"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["bic"])
    if df.empty:
        raise ValueError("No valid BIC values provided.")

    min_bic = df["bic"].min()
    df["delta_bic"] = df["bic"] - min_bic
    rel_likelihood = np.exp(-0.5 * df["delta_bic"])
    df["weight"] = rel_likelihood / rel_likelihood.sum()

    def interpret(delta):
        if delta < 2:
            return "weak"
        elif delta < 6:
            return "moderate"
        elif delta < threshold_strong:
            return "strong"
        else:
            return "decisive"

    df["evidence"] = df["delta_bic"].apply(interpret)
    return df.sort_values("bic", ascending=True).reset_index(drop=True)


# ============================================================
#  SYNTHETIC MOCKS FOR TESTING
# ============================================================
def synthetic_alt_bics(bic_null, seed=0, bias_strength=-2.0):
    """
    Generate mock alternative BICs for testing.
    """
    rng = np.random.default_rng(seed)
    return {
        "NGA": bic_null + rng.normal(bias_strength, 0.8),
        "radiation_pressure": bic_null + rng.normal(1.2, 0.6),
        "outgassing": bic_null + rng.normal(0.5, 0.8),
        "coherence": bic_null + rng.normal(bias_strength * 1.2, 0.7),
    }


# ============================================================
#  PHYSICAL ACTIVITY MODEL FITS
# ============================================================
def _comet_law(r, A0, m):
    """Classic cometary law A(r) = A0 * r^(-m)."""
    return A0 * np.power(np.clip(r, 1e-6, None), -m)


def _impulsive_jet(t, A_peak, t0, width):
    """Gaussian-shaped impulsive jet."""
    return A_peak * np.exp(-0.5 * ((t - t0) / width) ** 2)


def fit_activity_models(r, A_abs, t=None, seed=0):
    """
    Fit comet-law and impulsive-jet models to magnitude of acceleration.

    Parameters
    ----------
    r : array_like
        Heliocentric distances [au].
    A_abs : array_like
        Magnitude of non-gravitational acceleration.
    t : array_like or None
        Times (e.g., JD or relative days). Needed for jet model.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {'comet_law': bic_val, 'jet': bic_val, 'best': best_model_str}
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=float)
    A_abs = np.asarray(A_abs, dtype=float)
    mask = np.isfinite(r) & np.isfinite(A_abs) & (A_abs > 0)
    if np.count_nonzero(mask) < 5:
        return {"comet_law": np.nan, "jet": np.nan, "best": None}

    r, A_abs = r[mask], A_abs[mask]

    # --- Comet-law fit ---
    try:
        popt1, _ = curve_fit(_comet_law, r, A_abs, p0=[np.median(A_abs), 2.0])
        resid1 = A_abs - _comet_law(r, *popt1)
        bic1 = len(r) * np.log(np.var(resid1)) + 2 * np.log(len(r))
    except Exception:
        bic1 = np.nan

    # --- Jet fit (if time given) ---
    bic2 = np.nan
    if t is not None:
        t = np.asarray(t, dtype=float)
        t_norm = (t - np.nanmean(t)) / np.nanstd(t)
        try:
            p0 = [np.max(A_abs), 0.0, 0.5]
            popt2, _ = curve_fit(_impulsive_jet, t_norm, A_abs, p0=p0)
            resid2 = A_abs - _impulsive_jet(t_norm, *popt2)
            bic2 = len(r) * np.log(np.var(resid2)) + 3 * np.log(len(r))
        except Exception:
            bic2 = np.nan

    results = {"comet_law": bic1, "jet": bic2}
    df = compare_models(min(bic1, bic2), results)
    best = df.iloc[0]["model"] if not df.empty else None
    return {"comet_law": bic1, "jet": bic2, "best": best}


# ============================================================
#  GET BEST MODEL ENTRY
# ============================================================
def best_model(df_or_dict):
    """
    Identify the best-supported model entry from comparison output.
    """
    if isinstance(df_or_dict, dict):
        df = compare_models(
            df_or_dict.get("gravity_only", np.nan),
            {k: v for k, v in df_or_dict.items() if k != "gravity_only"},
        )
    else:
        df = df_or_dict.copy()

    if df.empty:
        raise ValueError("No valid BIC data provided.")

    top = df.iloc[0]
    return {
        "model": str(top["model"]),
        "bic": float(top["bic"]),
        "delta_bic": float(top["delta_bic"]),
        "weight": float(top["weight"]),
        "evidence": str(top["evidence"]),
    }
