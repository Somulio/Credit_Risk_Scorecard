"""Simplified IFRS 9 / Ind AS 109 Expected Credit Loss (ECL) computation."""
import numpy as np
import pandas as pd


def stage_account(dpd: int, sicr: bool = False) -> int:
    """IRAC-aligned staging: Stage 1 (<=30 DPD), Stage 2 (31-90 DPD or SICR), Stage 3 (90+ DPD/NPA)."""
    if dpd > 90:
        return 3
    if dpd > 30 or sicr:
        return 2
    return 1


def compute_ecl(pd_12m: pd.Series, lgd: pd.Series, ead: pd.Series,
                 stage: pd.Series, pd_lifetime: pd.Series = None) -> pd.Series:
    """
    Stage 1 -> 12-month ECL = PD_12m * LGD * EAD
    Stage 2/3 -> Lifetime ECL = PD_lifetime * LGD * EAD
    """
    if pd_lifetime is None:
        pd_lifetime = 1 - (1 - pd_12m) ** 3  # crude 3-yr lifetime approximation
    ecl_12m = pd_12m * lgd * ead
    ecl_lifetime = pd_lifetime * lgd * ead
    return np.where(stage == 1, ecl_12m, ecl_lifetime)
