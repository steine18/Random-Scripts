"""
ft2_discharge_sensitivity.py
────────────────────────────
Load a SonTek FlowTracker2 .ft file (or an unpacked directory / loose files)
and re-compute discharge using truncated averaging windows to assess sensitivity
of the result to averaging duration.

Discharge is recomputed using the mid-section method:
    Q = Σ  V_i · W_i · D_i
where
    V_i  = mean X-velocity for station i over the first N samples
    D_i  = measured depth at station i
    W_i  = mid-section width  =  0.5 · (dist_to_prev_station + dist_to_next_station)
           (bank stations contribute half-width to their adjacent open-water station)

Usage
─────
    python ft2_discharge_sensitivity.py <path>

<path> can be:
    - A .ft file  (ZIP archive containing DataFile.json + PointMeasurements/)
    - A directory containing DataFile.json (and optionally PointMeasurements/ subdir
      or the PointMeasurement_*.json files at the same level)
    - A DataFile.json file directly
"""

from __future__ import annotations

import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

# ── Unit conversions (metric → English) ────────────────────────────────────────
M_TO_FT    = 3.28084
M2_TO_FT2  = 10.7639
M3S_TO_CFS = 35.3147
MS_TO_FPS  = 3.28084


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RawSamples:
    """Raw ADV velocity samples for one point measurement."""
    pm_id: str
    sampling_rate_hz: float
    vx: np.ndarray          # shape (N,), m/s, NaN where spiked/missing
    vy: np.ndarray          # shape (N,), m/s
    timestamps: list[str] = field(default_factory=list)   # ISO 8601 UTC, one per sample
    spike_indices: list[int] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.vx)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sampling_rate_hz

    def end_time(self, n_samples: Optional[int] = None) -> Optional[str]:
        """
        UTC timestamp of the last sample in the averaging window.
        Returns None if no timestamps are available.
        """
        if not self.timestamps:
            return None
        idx = (n_samples - 1) if n_samples is not None else (len(self.timestamps) - 1)
        idx = min(idx, len(self.timestamps) - 1)
        return self.timestamps[idx]

    def mean_vx(self, n_samples: Optional[int] = None) -> float:
        """
        Mean X-velocity over the first n_samples samples.
        Spike indices within that window are replaced with NaN before averaging,
        matching FlowTracker2's despiking behaviour.
        If n_samples is None, use all samples.
        """
        vx = self.vx.copy()
        # Re-apply spike masking within requested window
        for idx in self.spike_indices:
            if idx < len(vx):
                vx[idx] = np.nan
        if n_samples is not None:
            vx = vx[:n_samples]
        valid = vx[~np.isnan(vx)]
        return float(np.mean(valid)) if len(valid) > 0 else np.nan


@dataclass
class Station:
    """One vertical in the cross section."""
    station_id: str
    location_m: float
    depth_m: float
    station_type: str           # RightBank | OpenWater | LeftBank | Ice | ...
    velocity_method: str        # SixTenths | TwoPoint | ThreePoint | ...
    correction_factor: float
    # Each station can have multiple point measurements (e.g. two-point method)
    point_measurement_ids: list[str] = field(default_factory=list)
    # Populated after loading PointMeasurement files
    raw_samples: list[RawSamples] = field(default_factory=list)
    # Width assigned by mid-section method (populated after geometry pass)
    width_m: Optional[float] = None

    @property
    def is_open_water(self) -> bool:
        return self.station_type == "OpenWater"

    def mean_vertical_velocity(self, n_samples: Optional[int] = None) -> float:
        """
        Mean velocity in the vertical, accounting for multi-point methods.
        Currently implements:
          SixTenths  → single point, direct mean
          TwoPoint   → mean of the two point means (0.2D + 0.8D) / 2
          ThreePoint → (0.2D + 0.6D + 0.8D) / 3  [equal weighting per USGS]
        Falls back to simple average of all point measurements if method unknown.
        Correction factor is applied at the end.
        """
        if not self.raw_samples:
            return np.nan

        method = self.velocity_method

        if method in ("SixTenths", "None") or len(self.raw_samples) == 1:
            v = self.raw_samples[0].mean_vx(n_samples)
        elif method == "TwoPoint":
            # (0.2D + 0.8D) / 2
            means = [pm.mean_vx(n_samples) for pm in self.raw_samples]
            v = float(np.nanmean(means))
        elif method == "ThreePoint":
            # Equal weights for 3-point method
            means = [pm.mean_vx(n_samples) for pm in self.raw_samples]
            v = float(np.nanmean(means))
        else:
            means = [pm.mean_vx(n_samples) for pm in self.raw_samples]
            v = float(np.nanmean(means))

        return v * self.correction_factor

    def discharge_contribution(self, n_samples: Optional[int] = None) -> float:
        """Q contribution for this station: V * W * D (mid-section method)."""
        if not self.is_open_water or self.width_m is None:
            return 0.0
        v = self.mean_vertical_velocity(n_samples)
        if np.isnan(v):
            return np.nan
        return v * self.width_m * self.depth_m


# ── Loader ────────────────────────────────────────────────────────────────────

class FT2Measurement:
    """Parsed FlowTracker2 measurement with full raw sample access."""

    def __init__(self, data_file: dict, point_measurements: dict[str, dict]):
        """
        Parameters
        ----------
        data_file : parsed DataFile.json
        point_measurements : {pm_id: parsed PointMeasurement_*.json}
        """
        self.raw_data_file = data_file
        self.properties = data_file.get("Properties", {})
        self.site_number = self.properties.get("SiteNumber", "")
        self.site_name = self.properties.get("SiteName", "")
        self.operator = self.properties.get("Operator", "")
        self.start_time = self.properties.get("StartTime", "")
        self.end_time = self.properties.get("EndTime", "")
        self.reported_discharge_m3s = (
            data_file.get("Calculations", {}).get("Discharge (m3/s)", np.nan)
        )
        self.reported_gage_height = (
            data_file.get("Calculations", {}).get("GaugeHeight", np.nan)
        )

        self.stations: list[Station] = []
        self._load_stations(data_file["Stations"], point_measurements)
        self._assign_mid_section_widths()

    # ── internal loaders ──────────────────────────────────────────────────────

    def _load_stations(
        self,
        station_dicts: list[dict],
        pm_lookup: dict[str, dict],
    ) -> None:
        for sd in station_dicts:
            pm_ids = [pm["Id"] for pm in sd.get("PointMeasurements", [])]
            st = Station(
                station_id=sd["Id"],
                location_m=sd["Location (m)"],
                depth_m=sd["Depth (m)"],
                station_type=sd["StationType"],
                velocity_method=sd.get("VelocityMethod", "SixTenths"),
                correction_factor=sd.get("CorrectionFactor", 1.0),
                point_measurement_ids=pm_ids,
            )
            for pm_id in pm_ids:
                pm_data = pm_lookup.get(pm_id)
                if pm_data is None:
                    print(f"  WARNING: PointMeasurement {pm_id} not found — skipping")
                    continue
                raw = _parse_point_measurement(pm_data)
                st.raw_samples.append(raw)
            self.stations.append(st)

    def _assign_mid_section_widths(self) -> None:
        """
        Mid-section width for each open-water station:
            W_i = 0.5 * (location[i+1] - location[i-1])
        Bank stations are included in the location list but get W=0.
        This matches the standard USGS/SonTek mid-section implementation.
        """
        locs = [s.location_m for s in self.stations]
        n = len(locs)
        for i, st in enumerate(self.stations):
            if not st.is_open_water:
                st.width_m = 0.0
                continue
            prev_loc = locs[i - 1] if i > 0 else locs[i]
            next_loc = locs[i + 1] if i < n - 1 else locs[i]
            st.width_m = 0.5 * (next_loc - prev_loc)

    # ── analysis ──────────────────────────────────────────────────────────────

    @property
    def sampling_rate_hz(self) -> float:
        for st in self.stations:
            if st.raw_samples:
                return st.raw_samples[0].sampling_rate_hz
        return 2.0

    @property
    def full_n_samples(self) -> int:
        """Number of samples in the full averaging window."""
        counts = [
            pm.n_samples
            for st in self.stations
            for pm in st.raw_samples
        ]
        return max(counts) if counts else 0

    def n_samples_for_duration(self, duration_s: float) -> int:
        """Convert a target duration to a sample count, clipped to available data."""
        n = int(round(duration_s * self.sampling_rate_hz))
        return min(n, self.full_n_samples)

    def compute_discharge(
        self, n_samples: Optional[int] = None
    ) -> tuple[float, pl.DataFrame]:
        """
        Re-compute total discharge using the first n_samples from each station.

        Returns
        -------
        total_q : float, m³/s
        detail  : DataFrame with per-station breakdown
        """
        rows = []
        for st in self.stations:
            if not st.is_open_water:
                continue
            v_ms = st.mean_vertical_velocity(n_samples)
            q_m3s = st.discharge_contribution(n_samples)
            rows.append({
                "location_ft": st.location_m * M_TO_FT,
                "depth_ft": st.depth_m * M_TO_FT,
                "width_ft": st.width_m * M_TO_FT,
                "velocity_fps": v_ms * MS_TO_FPS,
                "discharge_cfs": q_m3s * M3S_TO_CFS,
                "velocity_method": st.velocity_method,
                "n_point_measurements": len(st.raw_samples),
            })
        df = pl.DataFrame(rows)
        # Return total Q in m3/s internally for sensitivity calculations
        total_q_m3s = df["discharge_cfs"].sum() / M3S_TO_CFS
        return total_q_m3s, df

    def sensitivity_analysis(
        self, durations_s: list[float]
    ) -> pl.DataFrame:
        """
        Compute discharge for each duration in durations_s.

        Returns a DataFrame with one row per duration.
        """
        # Always include the full interval
        full_duration = self.full_n_samples / self.sampling_rate_hz
        all_durations = sorted(set(durations_s) | {full_duration})

        rows = []
        full_q = None
        for dur in all_durations:
            n = self.n_samples_for_duration(dur)
            actual_dur = n / self.sampling_rate_hz
            q_m3s, _ = self.compute_discharge(n_samples=n)
            if actual_dur == full_duration:
                full_q = q_m3s
            # Latest end timestamp across all stations for this window
            end_times = [
                pm.end_time(n)
                for st in self.stations if st.is_open_water
                for pm in st.raw_samples
            ]
            end_times = [t for t in end_times if t]
            end_time_utc = max(end_times) if end_times else None

            rows.append({
                "target_duration_s": float(dur),
                "actual_duration_s": actual_dur,
                "n_samples": n,
                "discharge_cfs": q_m3s * M3S_TO_CFS,
                "end_time_utc": end_time_utc,
            })

        df = pl.DataFrame(rows)
        full_cfs = df.filter(pl.col("actual_duration_s") == full_duration)["discharge_cfs"][0]
        df = df.with_columns(
            ((pl.col("discharge_cfs") - full_cfs) / full_cfs * 100.0)
            .alias("pct_change_from_full")
        )
        return df

    def station_velocity_sensitivity(
        self, durations_s: list[float]
    ) -> pl.DataFrame:
        """
        Per-station velocity for each duration — useful for seeing which
        stations drive discharge variability at short averaging windows.
        """
        full_duration = self.full_n_samples / self.sampling_rate_hz
        all_durations = sorted(set(durations_s) | {full_duration})

        rows = []
        for st in self.stations:
            if not st.is_open_water:
                continue
            row: dict = {
                "location_ft": st.location_m * M_TO_FT,
                "depth_ft": st.depth_m * M_TO_FT,
            }
            for dur in all_durations:
                n = self.n_samples_for_duration(dur)
                label = f"{int(dur)}s" if float(dur) == int(dur) else f"{dur}s"
                row[label] = st.mean_vertical_velocity(n) * MS_TO_FPS
            rows.append(row)

        return pl.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"FT2Measurement(site={self.site_number!r}, name={self.site_name!r}, "
            f"operator={self.operator!r}, start={self.start_time!r}, "
            f"n_stations={sum(1 for s in self.stations if s.is_open_water)} open-water)"
        )


# ── PointMeasurement parser ───────────────────────────────────────────────────

def _parse_point_measurement(pm: dict) -> RawSamples:
    """Extract raw Vx time series from a PointMeasurement JSON dict."""
    pm_id = pm.get("Id", "unknown")
    rate = pm.get("SamplingRate (Hz)", 2.0)
    samples = pm.get("Samples", [])

    vx_list = []
    for s in samples:
        v = s.get("Adv", {}).get("Velocity (m/s)", {})
        vx = v.get("X")
        vx_list.append(vx if vx is not None else np.nan)

    vx_arr = np.array(vx_list, dtype=float)

    # Spike indices from the top-level Spikes list
    # These are integer indices into the Samples array
    spike_indices = pm.get("Spikes", [])
    if isinstance(spike_indices, list):
        # Sometimes it's a list of ints, sometimes list of objects — handle both
        parsed_spikes = []
        for sp in spike_indices:
            if isinstance(sp, int):
                parsed_spikes.append(sp)
            elif isinstance(sp, dict):
                idx = sp.get("Index") or sp.get("SampleIndex")
                if idx is not None:
                    parsed_spikes.append(int(idx))
        spike_indices = parsed_spikes

    return RawSamples(
        pm_id=pm_id,
        sampling_rate_hz=float(rate),
        vx=vx_arr,
        vy=np.array(
            [
                (s.get("Adv", {}).get("Velocity (m/s)", {}).get("Y") or np.nan)
                for s in samples
            ],
            dtype=float,
        ),
        timestamps=[s.get("Time", "") for s in samples],
        spike_indices=spike_indices,
    )


# ── File loading ──────────────────────────────────────────────────────────────

def load_ft2(path: str | Path) -> FT2Measurement:
    """
    Load a FlowTracker2 measurement from:
      - a .ft file (ZIP)
      - a directory containing DataFile.json
      - a DataFile.json file directly

    The function searches for PointMeasurement_*.json files in the same
    directory or in a PointMeasurements/ subdirectory.
    """
    path = Path(path)

    if path.suffix.lower() == ".ft" or zipfile.is_zipfile(path):
        return _load_from_zip(path)
    elif path.is_dir():
        return _load_from_directory(path)
    elif path.name == "DataFile.json":
        return _load_from_directory(path.parent)
    else:
        raise ValueError(f"Cannot determine how to load: {path}")


def _load_from_zip(zip_path: Path) -> FT2Measurement:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # Find DataFile.json (may be at root or in a subdirectory)
        df_names = [n for n in names if n.endswith("DataFile.json")]
        if not df_names:
            raise FileNotFoundError("DataFile.json not found in archive")
        data_file = json.loads(zf.read(df_names[0]))

        pm_names = [n for n in names if "PointMeasurement_" in n and n.endswith(".json")]
        pm_lookup: dict[str, dict] = {}
        for pm_name in pm_names:
            pm_id = Path(pm_name).stem.replace("PointMeasurement_", "")
            pm_lookup[pm_id] = json.loads(zf.read(pm_name))

    return FT2Measurement(data_file, pm_lookup)


def _load_from_directory(directory: Path) -> FT2Measurement:
    df_path = directory / "DataFile.json"
    if not df_path.exists():
        raise FileNotFoundError(f"DataFile.json not found in {directory}")
    with open(df_path) as f:
        data_file = json.load(f)

    pm_lookup: dict[str, dict] = {}
    search_dirs = [directory, directory / "PointMeasurements"]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pm_path in search_dir.glob("PointMeasurement_*.json"):
            pm_id = pm_path.stem.replace("PointMeasurement_", "")
            with open(pm_path) as f:
                pm_lookup[pm_id] = json.load(f)

    return FT2Measurement(data_file, pm_lookup)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(m: FT2Measurement) -> None:
    print("=" * 60)
    print(f"Site:        {m.site_name} ({m.site_number})")
    print(f"Operator:    {m.operator}")
    print(f"Start:       {m.start_time}")
    print(f"End:         {m.end_time}")
    calcs = m.raw_data_file.get("Calculations", {})
    width  = calcs.get("Width (m)", np.nan)
    area   = calcs.get("Area (m2)", np.nan)
    depth  = calcs.get("Depth (m)", np.nan)
    vel    = calcs.get("Velocity (m/s)", {}).get("X", np.nan)
    temp   = calcs.get("Temperature (C)", np.nan)
    snr    = calcs.get("Snr (dB)", np.nan)
    rated  = calcs.get("RatedDischarge", np.nan)
    gh     = calcs.get("GaugeHeight", np.nan)
    print(f"Gage height: {gh:.2f} ft")
    print(f"Width:       {width * M_TO_FT:.2f} ft")
    print(f"Mean depth:  {depth * M_TO_FT:.3f} ft")
    print(f"Area:        {area * M2_TO_FT2:.3f} ft²")
    print(f"Mean vel:    {vel * MS_TO_FPS:.3f} ft/s")
    print(f"Temperature: {temp:.2f} °C")
    print(f"SNR:         {snr:.1f} dB")
    print(f"Rated Q:     {rated * M3S_TO_CFS:.2f} cfs")
    print(f"Reported Q:  {m.reported_discharge_m3s * M3S_TO_CFS:.2f} cfs")
    n_open = sum(1 for s in m.stations if s.is_open_water)
    rate   = m.sampling_rate_hz
    full_n = m.full_n_samples
    print(f"Stations:    {n_open} open-water verticals")
    print(f"Samples:     {full_n} per station @ {rate} Hz  ({full_n/rate:.0f} s)")
    print("=" * 60)


def print_sensitivity_table(sens: pl.DataFrame, reported_q_cfs: float) -> None:
    print("\nDischarge sensitivity to averaging duration")
    print("-" * 55)
    print(
        f"{'Target dur (s)':>14}  {'Actual (s)':>10}  {'N samples':>9}  "
        f"{'Q (cfs)':>9}  {'Δ from full (%)':>15}"
    )
    print("-" * 55)
    max_n = sens["n_samples"].max()
    for row in sens.iter_rows(named=True):
        marker = " ◄ reported" if row["n_samples"] == max_n else ""
        print(
            f"{row['target_duration_s']:>14.0f}  {row['actual_duration_s']:>10.1f}  "
            f"{row['n_samples']:>9d}  {row['discharge_cfs']:>9.3f}  "
            f"{row['pct_change_from_full']:>+14.2f}%"
            f"{marker}"
        )
    print("-" * 55)
    print(
        f"\nNote: reported Q = {reported_q_cfs:.3f} cfs "
        f"(re-computed full-window Q may differ slightly from instrument\n"
        f"      value due to floating-point summation order)"
    )



# ── Batch analysis ────────────────────────────────────────────────────────────

def analyze(
    path: "str | Path",
    durations_s: list = [10.0, 20.0, 40.0],
) -> dict:
    """
    Load a FlowTracker2 file and return a flat dict of metadata and discharge
    values at each requested averaging duration.

    Parameters
    ----------
    path : str or Path
        Path to a .ft file, directory, or DataFile.json.
    durations_s : list of float
        Averaging durations to compute discharge for. Defaults to [10, 20, 40].
        Durations longer than the recorded averaging interval are clipped to
        the full available window.

    Returns
    -------
    dict with keys:
        path                   : str
        site_number            : str
        site_name              : str
        operator               : str
        start_time             : str  ISO 8601 UTC
        date                   : str  YYYY-MM-DD local date
        n_stations             : int  open-water verticals only
        sampling_rate_hz       : float
        full_duration_s        : float  actual averaging duration used in field
        reported_discharge_cfs : float  as stored in the file
        q_{N}s_cfs             : float  one key per duration, e.g. q_10s_cfs
        pct_change_{N}s        : float  % vs full window, e.g. pct_change_10s
        error                  : str or None
    """
    path = Path(path)
    result = {
        "path": str(path),
        "site_number": None,
        "site_name": None,
        "operator": None,
        "start_time": None,
        "date": None,
        "n_stations": None,
        "sampling_rate_hz": None,
        "full_duration_s": None,
        "reported_discharge_cfs": None,
        "error": None,
    }

    try:
        m = load_ft2(path)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["site_number"] = m.site_number
    result["site_name"] = m.site_name
    result["operator"] = m.operator
    result["start_time"] = m.start_time
    result["n_stations"] = sum(1 for s in m.stations if s.is_open_water)
    result["sampling_rate_hz"] = m.sampling_rate_hz
    full_duration = m.full_n_samples / m.sampling_rate_hz
    result["full_duration_s"] = full_duration
    result["reported_discharge_cfs"] = m.reported_discharge_m3s * M3S_TO_CFS

    # Parse local date using the UTC offset stored in the file
    try:
        from datetime import datetime, timedelta
        dt_utc = datetime.fromisoformat(m.start_time.replace("Z", "+00:00"))
        offset_str = m.properties.get("LocalTimeUtcOffset", "+00:00")
        sign = 1 if offset_str[0] == "+" else -1
        h, mn, s = (int(x) for x in offset_str[1:].split(":"))
        offset = timedelta(hours=sign * h, minutes=sign * mn, seconds=sign * s)
        result["date"] = (dt_utc + offset).date().isoformat()
    except Exception:
        result["date"] = m.start_time[:10]

    # Discharge at each requested duration
    full_q, _ = m.compute_discharge()
    for dur in durations_s:
        n = m.n_samples_for_duration(dur)
        q_m3s, _ = m.compute_discharge(n_samples=n)
        label = f"{int(dur)}s" if float(dur) == int(dur) else f"{dur}s"
        result[f"q_{label}_cfs"] = q_m3s * M3S_TO_CFS
        result[f"pct_change_{label}"] = (
            100.0 * (q_m3s - full_q) / full_q if full_q else np.nan
        )
        # Latest end timestamp across all stations for this window
        end_times = [
            pm.end_time(n)
            for st in m.stations if st.is_open_water
            for pm in st.raw_samples
        ]
        result[f"end_time_utc_{label}"] = max((t for t in end_times if t), default=None)

    return result


def analyze_many(
    paths: list,
    durations_s: list = [10.0, 20.0, 40.0],
) -> pl.DataFrame:
    """
    Run analyze() on a list of .ft files and return a DataFrame, one row per file.
    Files that fail to load appear with their error column populated and null
    discharge values rather than raising.
    """
    rows = [analyze(p, durations_s=durations_s) for p in paths]
    return pl.DataFrame(rows)

# ── Entry point ───────────────────────────────────────────────────────────────

def main(path: str) -> FT2Measurement:
    print(f"\nLoading: {path}")
    m = load_ft2(path)
    print(f"Loaded:  {m}")

    print_summary(m)

    # Verify our re-computed Q matches the instrument's reported value
    full_q_m3s, station_detail = m.compute_discharge()
    full_q_cfs = full_q_m3s * M3S_TO_CFS
    reported_cfs = m.reported_discharge_m3s * M3S_TO_CFS
    print(
        f"\nRe-computed Q (full window): {full_q_cfs:.3f} cfs"
    )
    print(
        f"Instrument reported Q:       {reported_cfs:.3f} cfs"
    )
    diff_pct = 100 * (full_q_m3s - m.reported_discharge_m3s) / m.reported_discharge_m3s
    print(f"Difference:                  {diff_pct:+.4f}%")

    # Sensitivity analysis
    durations = [10.0, 20.0, 40.0]
    sens = m.sensitivity_analysis(durations)
    print_sensitivity_table(sens, reported_cfs)

    # Per-station velocity breakdown
    print("\nPer-station mean velocity (ft/s) by averaging duration")
    vel_df = m.station_velocity_sensitivity(durations)
    print(vel_df)

    return m   # return for interactive / notebook use
