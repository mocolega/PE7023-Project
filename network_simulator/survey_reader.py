"""
survey_reader.py
================
Read well survey CSV files and return a list of SurveyPoint objects.

Supported CSV formats
---------------------
The function auto-detects which columns are present:

  Format A – MD + TVD only (most common export from PIPESIM/Landmark):
      MD [ft], TVD [ft]
      Inclinations are back-calculated via the min-curvature inverse.

  Format B – MD + Inclination (standard directional survey):
      MD [ft], INC [deg]
      TVD is computed forward for reference only.

  Format C – MD + TVD + Inclination (full survey):
      MD [ft], TVD [ft], INC [deg]
      All three columns present — inclinations used directly, TVD verified.

Column name matching is case-insensitive and flexible:
  MD  : "md", "depth", "measured depth", "md_ft", "md (ft)"
  TVD : "tvd", "true vertical depth", "tvd_ft", "tvd (ft)"
  INC : "inc", "inclination", "incl", "angle", "inc_deg", "inc (deg)"

Usage
-----
    from survey_reader import read_survey
    survey = read_survey('W1_survey.csv')

    # Then use directly:
    well = Well(name="W1", fluid=fluid, survey=survey, ...)
"""

from __future__ import annotations
import csv
import math
import os
from typing import List, Optional

# Import SurveyPoint from network_components if available,
# otherwise define a local version for standalone use.
try:
    from network_components import SurveyPoint
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class SurveyPoint:
        md_ft:    float
        inc_deg:  float
        az_deg:   float = 0.0


# ── Column name aliases ──────────────────────────────────────────────────────

_MD_ALIASES  = {"md", "depth", "measured depth", "md_ft", "md (ft)",
                "measured_depth", "mddepth", "md[ft]", "md[m]",
                "md [ft]", "md [m]", "md(ft)", "md(m)"}

_TVD_ALIASES = {"tvd", "true vertical depth", "tvd_ft", "tvd (ft)",
                "true_vertical_depth", "tvddepth", "tvd[ft]", "tvd[m]",
                "tvd [ft]", "tvd [m]", "tvd(ft)", "tvd(m)"}

_INC_ALIASES = {"inc", "inclination", "incl", "angle", "inc_deg",
                "inc (deg)", "inc[deg]", "inclination (deg)",
                "deviation", "dip", "inc [deg]", "inc(deg)"}

_AZ_ALIASES  = {"az", "azimuth", "azi", "az_deg", "az (deg)",
                "az[deg]", "azimuth (deg)"}


def _match_col(header: str, aliases: set) -> bool:
    """Return True if header (normalised) is in the alias set."""
    return header.strip().lower() in aliases


def _find_col(headers: List[str], aliases: set) -> Optional[int]:
    """Return the index of the first matching column, or None."""
    for i, h in enumerate(headers):
        if _match_col(h, aliases):
            return i
    return None


# ── Min-curvature inverse: MD+TVD → inclinations ────────────────────────────

def _invert_md_tvd(md_vals: List[float],
                   tvd_vals: List[float]) -> List[float]:
    """
    Given paired (MD, TVD) stations, back-calculate inclination at each
    station using the min-curvature inverse.

    Assumes RF ≈ 1 per segment (small build-rate), giving:
        i_avg = acos(dTVD / dMD)
        i_end = 2 * i_avg - i_start   (symmetric split about average)

    Returns a list of inclinations [deg] the same length as md_vals.
    """
    inc = [0.0]   # surface always vertical

    for k in range(len(md_vals) - 1):
        dmd  = md_vals[k+1] - md_vals[k]
        dtvd = tvd_vals[k+1] - tvd_vals[k]

        if dmd <= 0:
            inc.append(inc[-1])
            continue

        cos_avg = max(-1.0, min(1.0, dtvd / dmd))
        i_avg   = math.degrees(math.acos(cos_avg))

        i_end = max(0.0, 2.0 * i_avg - inc[-1])
        inc.append(i_end)

    return inc


# ── Forward TVD from MD + inclination (for verification printout) ─────────

def _forward_tvd(md_vals: List[float],
                 inc_vals: List[float]) -> List[float]:
    """Compute TVD from MD + inclination using min-curvature."""
    tvd = [0.0]
    for k in range(len(md_vals) - 1):
        dmd = md_vals[k+1] - md_vals[k]
        i1  = math.radians(inc_vals[k])
        i2  = math.radians(inc_vals[k+1])
        beta = math.acos(max(-1.0, min(1.0, math.cos(i2 - i1))))
        rf   = 2 / beta * math.tan(beta / 2) if abs(beta) > 1e-9 else 1.0
        tvd.append(tvd[-1] + dmd / 2 * (math.cos(i1) + math.cos(i2)) * rf)
    return tvd


# ── Main public function ─────────────────────────────────────────────────────

def read_survey(filepath: str,
                delimiter: str = ',',
                verbose: bool = False) -> List[SurveyPoint]:
    """
    Read a well survey CSV and return a list of SurveyPoint objects.

    Parameters
    ----------
    filepath  : path to the CSV file
    delimiter : field separator (default ','; use '\\t' for TSV)
    verbose   : print a summary table after reading

    Returns
    -------
    List[SurveyPoint]  ready to pass directly to Well(survey=...)

    Raises
    ------
    FileNotFoundError  if the file does not exist
    ValueError         if neither MD+TVD nor MD+INC columns are found
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Survey file not found: '{filepath}'")

    # ── Read raw CSV ─────────────────────────────────────────────────────
    rows: List[List[str]] = []
    with open(filepath, newline='', encoding='utf-8-sig') as fh:
        # Auto-detect delimiter if caller passed 'auto'
        if delimiter == 'auto':
            sample = fh.read(2048)
            fh.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t;|')
            reader  = csv.reader(fh, dialect)
        else:
            reader = csv.reader(fh, delimiter=delimiter)

        for row in reader:
            if any(cell.strip() for cell in row):   # skip blank lines
                rows.append(row)

    if len(rows) < 2:
        raise ValueError(f"File '{filepath}' has fewer than 2 non-blank rows.")

    # ── Find header row (first row that contains at least one alias match) ─
    header_idx = 0
    headers: List[str] = []
    for i, row in enumerate(rows):
        normalised = [c.strip().lower() for c in row]
        has_md  = any(_match_col(c, _MD_ALIASES)  for c in normalised)
        has_tvd = any(_match_col(c, _TVD_ALIASES) for c in normalised)
        has_inc = any(_match_col(c, _INC_ALIASES) for c in normalised)
        if has_md and (has_tvd or has_inc):
            header_idx = i
            headers    = [c.strip() for c in row]
            break
    else:
        raise ValueError(
            f"Could not find a header row with MD and (TVD or INC) columns in "
            f"'{filepath}'.\nFirst row seen: {rows[0]}")

    # ── Locate column indices ─────────────────────────────────────────────
    col_md  = _find_col(headers, _MD_ALIASES)
    col_tvd = _find_col(headers, _TVD_ALIASES)
    col_inc = _find_col(headers, _INC_ALIASES)
    col_az  = _find_col(headers, _AZ_ALIASES)   # optional

    if col_md is None:
        raise ValueError("No MD column found.")
    if col_tvd is None and col_inc is None:
        raise ValueError("Need at least one of TVD or INC columns.")

    # ── Parse numeric data ────────────────────────────────────────────────
    md_vals:  List[float] = []
    tvd_vals: List[float] = []
    inc_vals: List[float] = []
    az_vals:  List[float] = []

    data_rows = rows[header_idx + 1:]
    for lineno, row in enumerate(data_rows, start=header_idx + 2):
        # Skip rows that are clearly comment / header repeats
        try:
            md = float(row[col_md].strip())
        except (ValueError, IndexError):
            continue

        md_vals.append(md)

        tvd = float(row[col_tvd].strip()) if col_tvd is not None else None
        inc = float(row[col_inc].strip()) if col_inc is not None else None
        az  = float(row[col_az].strip())  if col_az  is not None else 0.0

        tvd_vals.append(tvd)
        inc_vals.append(inc)
        az_vals.append(az)

    if len(md_vals) < 2:
        raise ValueError("Need at least 2 data rows after the header.")

    # ── Determine format and compute missing column ───────────────────────
    has_tvd = all(v is not None for v in tvd_vals)
    has_inc = all(v is not None for v in inc_vals)

    if has_tvd and has_inc:
        fmt = "C (MD + TVD + INC)"
        final_inc = [float(v) for v in inc_vals]
        final_tvd = [float(v) for v in tvd_vals]

    elif has_tvd and not has_inc:
        fmt = "A (MD + TVD → INC back-calculated)"
        tvd_f      = [float(v) for v in tvd_vals]
        final_inc  = _invert_md_tvd(md_vals, tvd_f)
        final_tvd  = tvd_f

    elif has_inc and not has_tvd:
        fmt = "B (MD + INC → TVD computed)"
        final_inc  = [float(v) for v in inc_vals]
        final_tvd  = _forward_tvd(md_vals, final_inc)

    else:
        raise ValueError("Mixed None/value in TVD and INC columns — check file.")

    # ── Build SurveyPoint list ────────────────────────────────────────────
    survey = [
        SurveyPoint(md_ft=md, inc_deg=inc, az_deg=az)
        for md, inc, az in zip(md_vals, final_inc, az_vals)
    ]

    # ── Verbose summary ───────────────────────────────────────────────────
    if verbose:
        name = os.path.basename(filepath)
        print(f"\n  Survey loaded: '{name}'  [{fmt}]  —  {len(survey)} stations")
        print(f"  {'MD (ft)':>9}  {'TVD (ft)':>9}  {'Inc (deg)':>10}", end="")
        if col_az:
            print(f"  {'Az (deg)':>9}", end="")
        print()
        print(f"  {'─'*32}")
        for i, sp in enumerate(survey):
            print(f"  {sp.md_ft:>9.1f}  {final_tvd[i]:>9.1f}  {sp.inc_deg:>10.3f}", end="")
            if col_az:
                print(f"  {sp.az_deg:>9.1f}", end="")
            print()
        tvd_td = final_tvd[-1]
        md_td  = md_vals[-1]
        print(f"  TD: MD={md_td:.0f} ft  TVD={tvd_td:.0f} ft  "
              f"Closure={md_td-tvd_td:.0f} ft")

    return survey
