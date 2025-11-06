import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .orbit_fit import fit_orbit_with_nga
from .orientation import orientation_metrics, plane_normals
from .models import compare_models, fit_activity_models
from .mc_resample import resample_and_fit


def make_figures(
    df_ast,
    out_dir="figures",
    results_dir="results",
    seed=0,
    object_name=None,
    force=False,
):
    """
    Generate all diagnostic figures and summary statistics for a single interstellar object.
    Includes extended model comparison (comet-law vs impulsive jet) and dual-plane metrics.
    """
    assert object_name is not None, "object_name must be specified."

    # -----------------------------------------------------------
    # Setup
    # -----------------------------------------------------------
    SEEDS = {"1I": 101, "2I": 202, "3I": 303}
    seed = SEEDS.get(object_name, seed)
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    expected_figs = [
        f"{object_name}_F1_orientation.png",
        f"{object_name}_F2_rolling.png",
        f"{object_name}_F3_injection.png",
        f"{object_name}_F4_models.png",
        f"{object_name}_F5_activity.png",
    ]
    if not force and all(os.path.exists(os.path.join(out_dir, f)) for f in expected_figs):
        print(f"[{object_name}] Figures already exist — skipping regeneration.")
        summary_path = os.path.join(results_dir, f"{object_name}_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                return json.load(f)
        force = True

    # -----------------------------------------------------------
    # Step 1 — Baseline orbital fits
    # -----------------------------------------------------------
    null = fit_orbit_with_nga(df_ast, with_nga=False, seed=seed)
    nga = fit_orbit_with_nga(df_ast, with_nga=True, seed=seed + 1)
    Avec = np.array([nga["A1"], nga["A2"], nga["A3"]])
    metrics = orientation_metrics(Avec)

    # Ensure outward-facing sign convention
    rhat = np.mean(df_ast[["x_au", "y_au", "z_au"]].to_numpy(), axis=0)
    rhat /= np.linalg.norm(rhat)
    if np.dot(Avec, rhat) < 0:
        Avec *= -1
        nga.update({"A1": float(Avec[0]), "A2": float(Avec[1]), "A3": float(Avec[2])})

    with open(os.path.join(results_dir, f"{object_name}_fit_results.json"), "w") as f:
        json.dump({"null": null, "nga": nga, "orientation": metrics}, f, indent=2)

    # -----------------------------------------------------------
    # Step 2 — Monte Carlo resampling for orientation scatter
    # -----------------------------------------------------------
    n_ecl, _ = plane_normals()
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.cross(n_ecl, ex)
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, n_ecl)
    ex /= np.linalg.norm(ex)

    A_samples = resample_and_fit(
        n_samples=800,
        fit_fn=fit_orbit_with_nga,
        df_astrometry=df_ast,
        seed=seed + 2,
        object_name=object_name,
        verbose=False,
    )

    proj = A_samples - (A_samples @ n_ecl)[:, None] * n_ecl[None, :]
    angles = np.arctan2(proj @ ey, proj @ ex)
    counts, edges = np.histogram(angles, bins=36, range=(-np.pi, np.pi))

    plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, projection="polar")
    ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges),
           align="center", color="tab:blue", alpha=0.75)
    ax.set_title(f"F1: Orientation around ecliptic normal ({object_name})", va="bottom")
    plt.savefig(os.path.join(out_dir, f"{object_name}_F1_orientation.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Rayleigh test for alignment
    C, S = np.mean(np.cos(angles)), np.mean(np.sin(angles))
    R = np.hypot(C, S)
    n = len(angles)
    p_rayleigh = np.exp(-n * R**2)

    # -----------------------------------------------------------
    # Step 3 — Rolling NGA components (R, T, N)
    # -----------------------------------------------------------
    jd = df_ast["JD_TDB"].to_numpy()
    jd_min, jd_max = jd.min(), jd.max()
    window_days, step_days = 40, 10

    A_r_t, A_t_t, A_n_t, t_centers = [], [], [], []
    for i, t in enumerate(np.arange(jd_min, jd_max - window_days, step_days)):
        mask = (jd >= t) & (jd < t + window_days)
        df_win = df_ast.loc[mask]
        if len(df_win) < 3:
            continue
        fit = fit_orbit_with_nga(df_win, with_nga=True, seed=seed + i)
        Avec = np.array([fit["A1"], fit["A2"], fit["A3"]])

        # Local orbital frame
        mid_jd = t + window_days / 2
        mid_idx = np.argmin(np.abs(jd - mid_jd))
        rvec = df_ast.loc[mid_idx, ["x_au", "y_au", "z_au"]].to_numpy()
        vvec = df_ast.loc[mid_idx, ["vx_au_d", "vy_au_d", "vz_au_d"]].to_numpy()
        rhat = rvec / np.linalg.norm(rvec)
        vhat = vvec / np.linalg.norm(vvec)
        nvec = np.cross(rhat, vhat)
        if np.linalg.norm(nvec) < 1e-12:
            continue
        nvec /= np.linalg.norm(nvec)
        that = np.cross(nvec, rhat)
        that /= np.linalg.norm(that)

        A_r_t.append(np.dot(Avec, rhat))
        A_t_t.append(np.dot(Avec, that))
        A_n_t.append(np.dot(Avec, nvec))
        t_centers.append(mid_jd)

    from scipy.ndimage import gaussian_filter1d
    A_r_t = gaussian_filter1d(A_r_t, sigma=1)
    A_t_t = gaussian_filter1d(A_t_t, sigma=1)
    A_n_t = gaussian_filter1d(A_n_t, sigma=1)

    plt.figure()
    plt.plot(t_centers, A_r_t, label="Radial (R)")
    plt.plot(t_centers, A_t_t, label="Transverse (T)")
    plt.plot(t_centers, A_n_t, label="Normal (N)")
    if "r_au" in df_ast.columns:
        peri_jd = df_ast.loc[df_ast["r_au"].idxmin(), "JD_TDB"]
        plt.axvline(peri_jd, color="gray", ls="--", alpha=0.5, label="Perihelion")
    plt.xlabel("Julian Date (TDB)")
    plt.ylabel("Acceleration (au d$^{-2}$)")
    plt.legend()
    plt.title(f"F2: Rolling NGA components — {object_name}")
    plt.savefig(os.path.join(out_dir, f"{object_name}_F2_rolling.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------
    # Step 4 — Injection–recovery (unchanged)
    # -----------------------------------------------------------
    # ... [your injection–recovery code block remains the same here]
    # (keep this unchanged — it already works perfectly)

    # -----------------------------------------------------------
    # Step 5 — Basic model comparison
    # -----------------------------------------------------------
    alt_models = {"NGA": nga["bic"]}
    bic_table = compare_models(null["bic"], alt_models)
    bic_table["delta_bic_pos"] = null["bic"] - bic_table["bic"]
    bic_table["delta_bic_log"] = np.sign(bic_table["delta_bic_pos"]) * np.log10(
        1 + np.abs(bic_table["delta_bic_pos"])
    )

    plt.figure(figsize=(5, 4))
    plt.bar(bic_table["model"], bic_table["delta_bic_log"], color="tab:green")
    plt.axhline(0, color="gray", ls="--")
    plt.ylabel("log₁₀(1 + ΔBIC)  (positive = better NGA)")
    plt.title(f"F4: Model comparison ({object_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{object_name}_F4_models.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------
    # Step 6 — Derived metrics (before summary)
    # -----------------------------------------------------------
    try:
        A_samples = np.asarray(A_samples)
        has_samples = A_samples.size > 0 and A_samples.ndim == 2 and A_samples.shape[1] == 3
    except Exception:
        has_samples = False

    # Mean acceleration vector
    if has_samples:
        Avec_mean = np.mean(A_samples, axis=0)
        mean_abs_A = float(np.mean(np.linalg.norm(A_samples, axis=1)))
    else:
        Avec_mean = np.array([nga.get("A1", 0.0), nga.get("A2", 0.0), nga.get("A3", 0.0)], dtype=float)
        mean_abs_A = float(np.linalg.norm(Avec_mean))

    # Radial alignment fraction
    rvec_mean = np.mean(df_ast[["x_au", "y_au", "z_au"]].to_numpy(), axis=0)
    rnorm = np.linalg.norm(rvec_mean)
    Anorm = np.linalg.norm(Avec_mean)
    if rnorm > 0 and Anorm > 0:
        rhat_global = rvec_mean / rnorm
        Ahat_mean = Avec_mean / Anorm
        f_radial = float(abs(np.dot(Ahat_mean, rhat_global)))
    else:
        f_radial = float("nan")

    # -----------------------------------------------------------
    # Step 7 — Summary
    # -----------------------------------------------------------
    summary = {
        "object": object_name,
        "mean_abs_A": mean_abs_A,
        "theta_ecl_deg": metrics.get("theta_ecl_deg"),
        "orientation_metrics": metrics,
        "rayleigh_R": float(R),
        "rayleigh_p": float(p_rayleigh),
        "f_radial": f_radial,
        "bic_table": bic_table.to_dict(orient="records"),
        "best_model": bic_table.iloc[0].to_dict(),
        "provenance": {
            "n_rows": int(len(df_ast)),
            "jd_min": float(df_ast["JD_TDB"].min()),
            "jd_max": float(df_ast["JD_TDB"].max()),
            "r_mean": float(df_ast["r_au"].mean()),
        },
    }

    # -----------------------------------------------------------
    # Step 8 — Physical activity model comparison (log-scaled)
    # -----------------------------------------------------------
    if np.isfinite(nga.get("bic", np.nan)) and np.isfinite(null.get("bic", np.nan)):

        if len(A_r_t) >= 5:
            A_abs = np.sqrt(np.array(A_r_t)**2 + np.array(A_t_t)**2 + np.array(A_n_t)**2)
            r_vals = np.interp(t_centers, df_ast["JD_TDB"], df_ast["r_au"])
            t_vals = np.array(t_centers)
            act_results = fit_activity_models(r_vals, A_abs, t=t_vals)
        else:
            act_results = {"comet_law": np.nan, "jet": np.nan, "best": None}

        # Determine best model
        if np.isfinite(act_results.get("comet_law", np.nan)) and np.isfinite(act_results.get("jet", np.nan)) and act_results["jet"] < act_results["comet_law"]:
            best_activity = "jet"
        elif np.isfinite(act_results.get("comet_law", np.nan)):
            best_activity = "comet_law"
        else:
            best_activity = None
        act_results["best"] = best_activity

        summary["activity_models"] = act_results
        summary["best_activity_model"] = best_activity

        # Plot F5 (log-transformed BIC difference)
        plt.figure(figsize=(5, 4))
        if np.isfinite(act_results.get("comet_law", np.nan)) and np.isfinite(act_results.get("jet", np.nan)):
            # Transform so higher = better, consistent with F4
            bic_vals = np.array([act_results["comet_law"], act_results["jet"]])
            delta_bic = np.max(bic_vals) - bic_vals  # higher = better
            log_scores = np.log10(1 + delta_bic)

            plt.bar(["Comet-law", "Impulsive jet"], log_scores, color=["tab:blue", "tab:red"])
            plt.ylabel(r"$\log_{10}(1 + \Delta{\rm BIC})$ (positive = better)")
            plt.title(f"F5: Activity model comparison ({object_name})")
        else:
            plt.text(0.25, 0.5, "Insufficient data", fontsize=12, ha="center")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{object_name}_F5_activity.png"), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        summary["activity_models"] = {"comet_law": np.nan, "jet": np.nan, "best": None}
        summary["best_activity_model"] = None

    # -----------------------------------------------------------
    # Step 9 — Save summary
    # -----------------------------------------------------------
    with open(os.path.join(results_dir, f"{object_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    delta_bic = null["bic"] - nga["bic"]
    delta_bic_log = np.sign(delta_bic) * np.log10(1 + np.abs(delta_bic))
    print(f"[{object_name}] ✅ Figures saved under {out_dir}/")
    print(f"   ΔBIC(log₁₀) = {delta_bic_log:.3f}, Rayleigh p = {p_rayleigh:.3e}")
    print(f"   Activity model: {summary.get('best_activity_model', 'N/A')}")
    return summary
