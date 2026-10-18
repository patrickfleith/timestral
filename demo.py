"""Zero-shot forecasting of real sensor data with Mistral's Timestral models.

Loads the Kaggle dataset patrickfleith/temperature-humidity-pressure-illuminance
(readings about every 5 seconds) and averages them into 5-minute bins, each stamped with
the mean of the readings in the 5 minutes before it. It forecasts temperature four hours
ahead using humidity, pressure and light as past covariates. The forecast is made at ten
points in time. Each one has lux and pressure plotted underneath, and all of it is saved
as one figure, temperature_forecasts.png.

Usage (after installing timestral, see the install line for Colab):
    python demo.py                                  # Timestral-209M, on GPU if available
    python demo.py --model 5M                       # smallest checkpoint, fastest on CPU
    python demo.py --csv path/to/DATA-large.CSV     # where the Kaggle CSV is
"""

import argparse
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from timestral import TimestralForecaster

MODEL_SIZES = ("5M", "28M", "123M", "209M")
TARGET = "temperature"
COVARIATES = ("humidity", "pressure", "lux")
BIN_MINUTES = 5  # each step is the mean of the readings in the last BIN_MINUTES minutes
STEPS_PER_HOUR = 60 // BIN_MINUTES
CONTEXT_STEPS = 24 * STEPS_PER_HOUR  # one day of history, all of it fed to the model
HORIZON_HOURS = 4
HORIZON_STEPS = HORIZON_HOURS * STEPS_PER_HOUR  # forecast four hours ahead
PLOT_HISTORY_STEPS = 6 * STEPS_PER_HOUR  # history drawn under each forecast point
# Ten forecast points, as fractions of the record: the original four plus six more.
ORIGIN_FRACTIONS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# (lower level, upper level, label), widest first
BANDS = ((0.01, 0.99, "98% band"), (0.05, 0.95, "90% band"), (0.1, 0.9, "80% band"))
BAND_ALPHAS = (0.12, 0.2, 0.32)


def load_binned(csv_path: Path) -> pd.DataFrame:
    """Average the raw readings over the last BIN_MINUTES minutes, stamped at the end of each bin."""
    raw = pd.read_csv(csv_path)
    # Timestamps look like 2022/3/12 22:32:2 (seconds are not zero-padded).
    raw = raw.set_index(pd.to_datetime(raw.pop("time"), format="%Y/%m/%d %H:%M:%S")).sort_index()
    # closed="right", label="right": the stamp 22:35 is the mean of the readings in (22:30, 22:35].
    binned = raw.resample(f"{BIN_MINUTES}min", closed="right", label="right").mean()  # empty bins become NaN
    binned["pressure"] = binned["pressure"] / 100  # Pa -> hPa
    return binned


def pick_origins(binned: pd.DataFrame) -> list[int]:
    """Row positions of the forecast points: evenly spaced, with a fully observed target horizon."""
    target = binned[TARGET].to_numpy()
    origins = []
    for fraction in ORIGIN_FRACTIONS:
        i = max(int(len(target) * fraction), CONTEXT_STEPS)
        while np.isnan(target[i:i + HORIZON_STEPS]).any():
            i += 1
        origins.append(i)
    return origins


def plot_forecasts(path: Path, binned: pd.DataFrame, origins: list[int], median: np.ndarray,
                   bands: list, titles: list[str]) -> None:
    index = binned.index
    target = binned[TARGET].to_numpy()
    covariate_values = {name: binned[name].to_numpy() for name in ("lux", "pressure")}
    n_rows = -(-len(origins) // 2)  # two forecast points per row of the figure
    fig = plt.figure(figsize=(16, 4.6 * n_rows))
    grid = fig.add_gridspec(3 * n_rows, 2, height_ratios=[3, 1.3, 1.3] * n_rows,
                            top=0.95, bottom=0.03, left=0.06, right=0.98, hspace=0.5, wspace=0.18)
    for k, origin in enumerate(origins):
        row, col = divmod(k, 2)
        top_row = 3 * row
        hist_slice = slice(origin - PLOT_HISTORY_STEPS, origin)
        future_slice = slice(origin, origin + HORIZON_STEPS)
        t_hist = index[hist_slice].to_numpy()
        t_future = index[future_slice].to_numpy()

        ax = fig.add_subplot(grid[top_row, col])
        ax.plot(t_hist, target[hist_slice], color="0.3", lw=0.9, label="history")
        ax.plot(t_future, target[future_slice], color="0.3", lw=1.0, ls="--", label="actual")
        for (lo, hi), (_, _, label), alpha in zip(bands, BANDS, BAND_ALPHAS):
            ax.fill_between(t_future, lo[k], hi[k], color="tab:blue", alpha=alpha, lw=0, label=label)
        ax.plot(t_future, median[k], color="tab:blue", lw=1.8, label="median")
        ax.axvline(index[origin], color="0.5", lw=0.8, ls=":")
        ax.set_title(titles[k], loc="left", fontsize=10)
        ax.set_ylabel("temperature (°C)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.tick_params(labelbottom=False)

        # Lux and pressure are past covariates, so they are shown as observed, not forecast.
        for sub, (name, ylabel) in enumerate((("lux", "lux"), ("pressure", "pressure (hPa)")), start=1):
            sub_ax = fig.add_subplot(grid[top_row + sub, col], sharex=ax)
            values = covariate_values[name]
            sub_ax.plot(t_hist, values[hist_slice], color="0.3", lw=0.9)
            sub_ax.plot(t_future, values[future_slice], color="0.3", lw=1.0, ls="--")
            sub_ax.axvline(index[origin], color="0.5", lw=0.8, ls=":")
            sub_ax.set_ylabel(ylabel, fontsize=9)
            sub_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            sub_ax.grid(alpha=0.3)
            if sub == 1:
                sub_ax.tick_params(labelbottom=False)
    fig.suptitle(f"Timestral temperature forecasts, {HORIZON_HOURS} hours ahead ({BIN_MINUTES}-minute means). "
                 "Lux and pressure are past covariates, shown as observed.", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODEL_SIZES, default="209M", help="Timestral checkpoint size")
    parser.add_argument("--csv", type=Path, default=Path("DATA-large.CSV"), help="path to the Kaggle CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("plots"), help="directory for the PNG file")
    # parse_known_args: Colab and Jupyter add their own arguments to sys.argv when a cell runs
    args, _ = parser.parse_known_args()

    repo_id = f"mistralai/Timestral-{args.model}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    binned = load_binned(args.csv)
    print(f"{len(binned)} {BIN_MINUTES}-minute bins, {binned.index[0]} to {binned.index[-1]}; "
          f"{binned[TARGET].isna().sum()} bins without readings")
    origins = pick_origins(binned)

    print(f"Loading {repo_id} on {device} (the first run downloads the weights from the Hugging Face Hub)")
    start_time = time.perf_counter()
    forecaster = TimestralForecaster.from_pretrained(repo_id, device=device)
    print(f"Loaded in {time.perf_counter() - start_time:.1f}s")

    # All forecast points batched together: target shape (batch, 1, time), covariates (batch, 3, time).
    target = binned[TARGET].to_numpy()
    covariates = binned[list(COVARIATES)].to_numpy().T
    context = np.stack([target[o - CONTEXT_STEPS:o] for o in origins])[:, None, :]
    past = np.stack([covariates[:, o - CONTEXT_STEPS:o] for o in origins])
    actual = np.stack([target[o:o + HORIZON_STEPS] for o in origins])

    start_time = time.perf_counter()
    forecast = forecaster.forecast(context, prediction_length=HORIZON_STEPS, past_covariates=past)
    print(f"Forecast {HORIZON_HOURS} hours at {len(origins)} points in {time.perf_counter() - start_time:.2f}s")

    def numpy_of(tensor) -> np.ndarray:
        return tensor.cpu().numpy()[:, 0]  # drop the variate axis

    median = numpy_of(forecast.median)  # (points, horizon)
    bands = [(numpy_of(forecast.quantile(lo)), numpy_of(forecast.quantile(hi))) for lo, hi, _ in BANDS]

    titles = []
    print()
    print(f"{'forecast point':<18} {'MAE':>8} {'persistence':>12}")
    for k, origin in enumerate(origins):
        # Persistence baseline: repeat the last observed temperature.
        last_seen = target[:origin][np.isfinite(target[:origin])][-1]
        mae = np.mean(np.abs(median[k] - actual[k]))
        baseline_mae = np.mean(np.abs(last_seen - actual[k]))
        stamp = binned.index[origin]
        print(f"{stamp:%Y-%m-%d %H:%M} {mae:>8.3f} {baseline_mae:>12.3f}")
        titles.append(f"{stamp:%Y-%m-%d %H:%M}: MAE {mae:.3f} °C (persistence {baseline_mae:.3f} °C)")
    print(f"MAE in °C over the next {HORIZON_HOURS} hours; persistence repeats the last observed temperature")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.out_dir / "temperature_forecasts.png"
    plot_forecasts(fig_path, binned, origins, median, bands, titles)
    print(f"\nSaved {fig_path}")


if __name__ == "__main__":
    main()
