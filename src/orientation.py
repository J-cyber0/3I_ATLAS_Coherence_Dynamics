import numpy as np


# ----------------------------------------------------------------------
# Basic vector helpers
# ----------------------------------------------------------------------
def unit(v):
    """Return the normalized (unit) vector of v."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros_like(v)


def angle_between(u, v):
    """
    Compute the angle (degrees) between two 3D vectors.

    Parameters
    ----------
    u, v : array_like
        Input vectors.

    Returns
    -------
    float
        Angle in degrees (0–180).
    """
    u, v = unit(u), unit(v)
    c = np.clip(np.dot(u, v), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# ----------------------------------------------------------------------
# Reference planes
# ----------------------------------------------------------------------
def plane_normals():
    """
    Define reference plane normals for orientation analysis.

    Returns
    -------
    tuple of np.ndarray
        (n_ecliptic, n_jupiter)
    """
    # Ecliptic plane normal = z-axis
    n_ecl = np.array([0.0, 0.0, 1.0])

    # Jupiter Laplace plane — tilted ~1.3° in x–z plane
    tilt = np.radians(1.3)
    n_jup = np.array([np.sin(tilt), 0.0, np.cos(tilt)])

    return n_ecl, n_jup


# ----------------------------------------------------------------------
# Orientation metrics
# ----------------------------------------------------------------------
def orientation_metrics(Avec):
    """
    Compute angular relationships between the given vector (e.g. NGA)
    and reference planes (ecliptic, Jupiter Laplace).

    Parameters
    ----------
    Avec : array_like
        Vector [A1, A2, A3] or (N,3) array of sample vectors.

    Returns
    -------
    dict
        {
            "theta_ecl_deg": angle vs ecliptic normal,
            "theta_jup_deg": angle vs Jupiter-plane normal,
            "Cphi": coherence statistic = ½[cos(θ_ecl) + cos(θ_jup)]
        }
    """
    Avec = np.atleast_2d(Avec)
    n_ecl, n_jup = plane_normals()

    # Use mean direction for multi-sample input
    A_mean = np.mean(Avec, axis=0)
    theta_ecl = angle_between(A_mean, n_ecl)
    theta_jup = angle_between(A_mean, n_jup)

    Cphi = 0.5 * (np.cos(np.radians(theta_ecl)) + np.cos(np.radians(theta_jup)))

    return {
        "theta_ecl_deg": float(theta_ecl),
        "theta_jup_deg": float(theta_jup),
        "Cphi": float(Cphi),
    }


# ----------------------------------------------------------------------
# Probability tools
# ----------------------------------------------------------------------
def spherical_cap_pvalue(theta_deg):
    """
    Compute the one-sided spherical-cap probability that a random
    vector lies within angle <= θ of a given direction.

    P = (1 - cos θ) / 2

    Parameters
    ----------
    theta_deg : float
        Angle in degrees.

    Returns
    -------
    float
        Probability (0–1).
    """
    theta = np.radians(theta_deg)
    return 0.5 * (1.0 - np.cos(theta))
