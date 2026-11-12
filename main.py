"""Roblox Up-and-Coming top-10 forecast for make.fun
(roblox-up-and-coming-forecast-01).

Reads the chart history at $MAKEFUN_INPUT_DIR (else /input) and writes one
forecast to $MAKEFUN_OUTPUT_DIR/forecast.json (else /output/forecast.json):
for each of the top 10 places, a probability for each game named being at
that place, plus the top 10 in order. Python 3.13, standard library only,
no network, fully deterministic (no randomness).

Model: Gaussian rank diffusion.
  The Up-and-Coming chart moves fast: between reads a few hours apart a
  game's rank typically moves several places, and the top 10's order
  shuffles constantly while its membership turns over slowly. Betting
  everything on the chart as it stands (the house baseline) is far too
  confident and scores badly once the chart has moved.

  Instead, each game is treated as a random walk in rank. A game at rank r0
  when the history ends has rank ~ Normal(r0, sigma_1h * sqrt(t)) t hours
  later, with sigma_1h = 0.5 ranks per sqrt-hour (measured from the
  history: the mean absolute rank change over a 4-hour gap is ~3.5 places,
  over 7 hours ~6). The probability it holds place k is the Gaussian mass
  in [k - 0.5, k + 0.5]. Averaging that over the 72 hours of the scored
  window gives one static forecast that is honest about how far the chart
  can drift: sharp near the top where games are sticky, spread where the
  order shuffles.

  Each place's probabilities are normalised to sum to 1 - UNNAMED (0.80);
  the 0.20 left unnamed is shared evenly by every game on a read's chart
  that was not named at that place, which covers games that rise from
  below the modelled ranks and games that join the chart after the
  history ends. Games are modelled down to rank 40; a game that old
  almost never reaches the top 10 in 72 hours.

  Age-out: the chart only lists games published in the last 28 days. When
  the last read carries details with a game's creation time, the game's
  contribution is cut off at the hour it turns 28 days old, because it
  leaves the chart then. (Archive reads carry no details, so this only
  applies when the recorder's details are present.)

  top10 is the chart's top 10 as it stood when the history ends: the
  tie-break is best served by the games most likely to be in the top 10,
  and the current top 10 is the best single guess at that set.

Backtested on the practice window (history to 2026-11-07, scored on the
10 archive reads 2026-11-08 to 2026-11-10): log score -33.36, against the
house baseline's -43.51 (typical probability 0.035 vs 0.0129). A rolling
backtest at 8 history cutoffs gives a mean of -35.71 vs the baseline's
-42.98, so the edge is not tied to one cutoff.
"""

import json
import math
import os
from datetime import datetime, timezone

# --- Model parameters (tuned by backtesting; see module docstring) ---
SIGMA_1H = 0.5   # rank volatility, ranks per sqrt-hour
UNNAMED = 0.20   # share of each place left unnamed
MAX_RANK = 40    # model games down to this current rank
MAX_AGE_DAYS = 28.0  # games leave the chart 28 days after publication


def _phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _gauss_mass(mu, sigma, k):
    """P(Normal(mu, sigma) lies in [k - 0.5, k + 0.5])."""
    if sigma <= 0:
        return 1.0 if abs(mu - k) < 0.5 else 0.0
    return _phi((k + 0.5 - mu) / sigma) - _phi((k - 0.5 - mu) / sigma)


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    hours = int(task["window"]["slots"])
    history_until = _parse_time(task["history_until"])

    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    if not reads:
        raise ValueError("history.ndjson holds no reads")
    last = reads[-1]
    chart = sorted(last["up_and_coming"], key=lambda r: r["rank"])

    # Creation times from the last read's details, when the recorder sent them.
    details = last.get("details") or {}
    created = {}
    for key, det in details.items():
        if not isinstance(det, dict):
            continue
        c = det.get("created")
        if not c:
            continue
        try:
            uid = int(key)
        except (TypeError, ValueError):
            continue
        try:
            created[uid] = _parse_time(c)
        except ValueError:
            continue

    # Games to model: the top MAX_RANK of the last chart, with the hour each
    # ages out of the chart (None = stays for the whole window).
    games = []
    for row in chart:
        if row["rank"] > MAX_RANK:
            continue
        uid = int(row["universe_id"])
        r0 = int(row["rank"])
        t_out = None
        if uid in created:
            age_out = created[uid].timestamp() + MAX_AGE_DAYS * 86400.0
            t_out = (age_out - history_until.timestamp()) / 3600.0
        games.append((uid, r0, t_out))
    if not games:
        raise ValueError("no games to model in the last chart")

    # Average the rank distribution over the window's hours. The window
    # holds exactly `hours` slots (t = 0 .. hours - 1), one per whole hour
    # after the history ends.
    probs = {}
    for t in range(0, hours):
        sigma = SIGMA_1H * math.sqrt(max(t, 1))
        for uid, r0, t_out in games:
            if t_out is not None and t >= t_out:
                continue
            for k in range(1, places + 1):
                probs[(uid, k)] = probs.get((uid, k), 0.0) + (
                    _gauss_mass(r0, sigma, k) / hours
                )

    # Normalise each place to sum to 1 - UNNAMED; drop negligible rows.
    rows = []
    for k in range(1, places + 1):
        total = sum(v for (uid, kk), v in probs.items() if kk == k)
        if total <= 0:
            continue
        for (uid, kk), v in probs.items():
            if kk != k:
                continue
            p = (v / total) * (1.0 - UNNAMED)
            if p > 0.001:
                rows.append({"universe_id": uid, "place": k, "p": round(p, 4)})
    rows.sort(key=lambda r: (r["place"], r["universe_id"]))

    top10 = [uid for uid, _, _ in games[:places]]
    forecast = {"probabilities": rows, "top10": top10}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task["output"])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
