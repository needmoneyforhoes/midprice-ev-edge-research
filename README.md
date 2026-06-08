# midprice-ev-edge-research

Offline backtest that measures conditional EV for Polymarket mid-price ($0.40-$0.50) against-crowd entries on 5-minute BTC/XRP up/down markets.

A side's ask at $0.45 implies a 45% breakeven win rate (near coin-flip). This harness replays recorded tick histories, fires a fixed-share entry on the first tick where a side's ask enters the band, holds to settlement, and tests whether any signal-conditioned cut clears breakeven. No live trading.

## Scripts

- `research_dn_midprice.py`: full pipeline. Loads markets, runs the single-signal sweeps, grid search, sub-range and extended-range scans, prints a report.

Stages inside the script:

- `load_markets` / `find_first_tick_in_range`: load `market_history.jsonl`, find the first tick where `dn_ask`/`up_ask` lands in the band.
- `compute_pnl`: held-to-settle PnL per fire; bumps share count to the $1.00 Polymarket order minimum.
- `stats` / `bootstrap_ci`: t-test, Sharpe, 2000-sample bootstrap 95% CI. Marks CI-excludes-zero results.
- Signal sweeps A-F: univariate cuts on time remaining (`cd`), Binance momentum (`bn_delta_pct`), 10s price momentum, centerline regime (`cl_delta`), crowd conviction, and depth strength ratio (`*_sr`).
- Grid search: all cd x bn x d10s x cl x conv combos, max 3 active filters, min 12 fires, EV+ only, ranked by Sharpe x sqrt(n).
- Range scans: $0.05-$0.60 in 5c bands plus sub-ranges within $0.40-$0.50, run for DN and UP.

Output ends with a decision rule: CI-excludes-zero goes to shadow then live; otherwise the zone is left alone.

## Requirements

Python 3, standard library only. No third-party deps, no virtualenv.

## Usage

```bash
python3 research_dn_midprice.py market_history.jsonl
# falls back to ./market_history.jsonl if no path is given
```

## Data

Reads one newline-delimited JSON file, one market per line with `ticks`, `tick_columns`, and a `winner` of `UP`/`DN`. Not committed (see `.gitignore`); fetch it from the private polymarket-data repo.

Read-only; no wallet, key, or CLOB access.
