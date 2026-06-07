# midprice-ev-edge-research

Positive-EV edge research for Polymarket mid-price against-crowd entries, with grids over BN/momentum/conviction/depth.

## Why it exists

On Polymarket 5-minute BTC/XRP up/down markets, the cheap "lottery" zone ($0.05–$0.15) and late-conviction plays are already well-mined. This repo answers a harder question: **when a side's ask sits in the $0.40–$0.50 mid zone (a near-coin-flip where you are betting *against* the crowd), is there a signal-conditioned entry with positive expected value?** It is an offline backtest/edge-discovery harness, not a live trader.

## How it works

A single self-contained script, `research_dn_midprice.py`. It replays recorded per-market tick histories, finds the first tick where a side's ask enters the target band, simulates a fixed-share entry held to settlement, and measures conditional EV against the $0.45 breakeven (45% WR).

| Stage | What it does |
|-------|--------------|
| `load_markets` / `find_first_tick_in_range` | Loads `market_history.jsonl`, locates the first tick where `dn_ask`/`up_ask` falls in the price band |
| `compute_pnl` | Held-to-settle PnL per fire; bumps share count to the $1.00 Polymarket order minimum |
| `stats` / `bootstrap_ci` | t-test, Sharpe, and 2000-sample bootstrap 95% CI; flags CI-excludes-zero with `★★★` |
| Signal sweeps A–F | Univariate cuts on time-remaining (`cd`), Binance momentum (`bn_delta_pct`), 10s price momentum, centerline regime (`cl_delta`), crowd conviction, and depth strength ratio (`*_sr`) |
| Grid search | All cd × bn × Δ10s × cl × conv combos (max 3 active filters), ranked by Sharpe·√n, EV+ only |
| Extended range scan | Sweeps $0.05–$0.60 in 5¢ bands and sub-ranges within $0.40–$0.50, run symmetrically for DN and UP |

Output is a printed report ending in a deploy/SHADOW decision framework (CI-excludes-zero → shadow then live; else don't trade the zone).

## Requirements

- **Python 3** — standard library only (`json`, `math`, `random`, `statistics`-free). No third-party pip deps, no virtualenv required.
- **Data**: the recorded market history (see below). No wallet, private key, or network access — this script never touches funds or the CLOB.

## Usage

```bash
python3 research_dn_midprice.py market_history.jsonl
# defaults to ./market_history.jsonl if no argument is given
```

## Data

The script reads a single newline-delimited JSON file (default `market_history.jsonl`), one market per line with `ticks`, `tick_columns`, and a `winner` of `UP`/`DN`. This file is **not committed** (see `.gitignore`); fetch it from the private **polymarket-data** repo and pass its path as the first argument or place it in the working directory.

> Private research software. No warranty; trades/handles real funds at your own risk.
