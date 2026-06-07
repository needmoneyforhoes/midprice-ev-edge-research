#!/usr/bin/env python3
"""
RESEARCH: DN mid-price strategy ($0.40-$0.50 entry) profitability
================================================================
Run on VPS:  python3 research_dn_midprice.py market_history.jsonl

Question: In markets where DN never crashed to lottery ($0.05-$0.10),
but cheapest DN ask was $0.40-$0.50 — is there a signal-conditioned
DN-buy strategy with positive EV?

Math:
  DN ask = $0.45 → breakeven WR = 45%
  DN base rate = ~52.9% → blind EV = +$0.079/share
  But CONDITIONAL WR when DN is at $0.45 may differ from base rate.
  Key: DN at $0.45 means crowd says 55% UP. We're betting AGAINST crowd.
  Need a signal that tells us when crowd is WRONG at this price level.

Signals tested:
  A: cd (time remaining) — early vs late entry
  B: bn_delta_pct (Binance momentum) — with/against BTC trend  
  C: Δ10s_dn (recent DN momentum) — DN recovering or collapsing
  D: cl_delta (CL centerline) — regime gate alignment
  E: conv (crowd conviction) — how decisive is the crowd
  F: dn_sr (DN depth strength ratio)
  G: Combined grid search for best 2-3 feature filter

Also runs SYMMETRIC test: UP mid-price ($0.40-$0.50) for completeness.
"""
import json, sys, math, random, os
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────
DN_LO, DN_HI = 0.40, 0.50          # target DN ask range
UP_LO, UP_HI = 0.40, 0.50          # symmetric UP ask range  
MIN_SAMPLE   = 12                   # minimum fires for statistical test
BOOTSTRAP_N  = 2000
SEED         = 42
PM_MIN_COST  = 1.00                 # Polymarket minimum order
DEFAULT_SH   = 5                    # default share count

# ─── Helpers ──────────────────────────────────────────────────────
def load_markets(path):
    markets = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                if d.get("ticks") and d.get("winner") in ("UP", "DN"):
                    markets.append(d)
            except Exception:
                pass
    markets.sort(key=lambda m: m.get("ts", ""))
    return markets

def col_idx(market):
    return {c: i for i, c in enumerate(market["tick_columns"])}

def safe(tick, cols, key, default=0):
    i = cols.get(key)
    if i is None:
        return default
    v = tick[i]
    return v if v is not None else default

def compute_pnl(winner, ask_price, shares=DEFAULT_SH, side="DN", bankroll=25.0):
    """PnL for buying `side` at ask_price. Bumps to PM minimum if needed."""
    cost = shares * ask_price
    if cost < PM_MIN_COST:
        shares = max(shares, math.ceil(PM_MIN_COST / ask_price))
        cost = shares * ask_price
    if winner == side:
        return shares * (1.0 - ask_price)   # win: $1 - entry per share
    else:
        return -cost                         # loss: lose entire cost

def stats(pnls):
    n = len(pnls)
    if n == 0:
        return {"n": 0, "wr": 0, "mean": 0, "sd": 0, "total": 0,
                "t": 0, "p": 1.0, "sharpe": 0}
    mean = sum(pnls) / n
    wins = sum(1 for x in pnls if x > 0)
    if n > 1:
        var = sum((x - mean)**2 for x in pnls) / (n - 1)
        sd = math.sqrt(var)
        se = sd / math.sqrt(n)
        t = mean / se if se > 0 else 0
        # Normal approx p-value (two-tailed)
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if abs(t) < 10 else 0.0
    else:
        sd = 0; t = 0; p = 1.0
    return {"n": n, "wr": wins / n, "mean": mean, "sd": sd,
            "total": sum(pnls), "t": t, "p": p,
            "sharpe": mean / sd if sd > 0 else 0}

def bootstrap_ci(pnls, alpha=0.05):
    if len(pnls) < 3:
        return (None, None)
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOTSTRAP_N):
        sample = [rng.choice(pnls) for _ in range(len(pnls))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(BOOTSTRAP_N * alpha / 2)]
    hi = means[int(BOOTSTRAP_N * (1 - alpha / 2))]
    return (lo, hi)

def fmt_row(label, pnls, max_label=55):
    s = stats(pnls)
    ci = bootstrap_ci(pnls)
    ci_str = f"[${ci[0]:+.3f}, ${ci[1]:+.3f}]" if ci[0] is not None else "—"
    sig = ""
    if s["mean"] > 0 and ci[0] is not None and ci[0] > 0:
        sig = " ★★★"  # CI excludes zero
    elif s["mean"] > 0 and s["p"] < 0.10:
        sig = " ★★"
    elif s["mean"] > 0:
        sig = " ★"
    elif s["mean"] < 0:
        sig = " ✗"
    return (f"  {label:<{max_label}}{s['n']:<6}{s['wr']*100:<7.1f}%"
            f"${s['total']:<+11.2f}${s['mean']:<+10.4f}"
            f"{s['sharpe']:<+9.3f}{s['p']:<9.4f}{ci_str}{sig}")

HEADER = (f"  {'Filter':<55}{'n':<6}{'WR':<8}{'Total':<12}{'μ/fire':<11}"
          f"{'Sharpe':<9}{'p':<9}{'95% CI':<24}")
SEP = "  " + "─" * 130

# ─── Find first tick in price range ───────────────────────────────
def find_first_tick_in_range(market, side, lo, hi):
    """Return (tick_index, tick, cols) for first tick where side's ask ∈ [lo, hi]."""
    cols = col_idx(market)
    ask_key = "dn_ask" if side == "DN" else "up_ask"
    for i, t in enumerate(market["ticks"]):
        ask = safe(t, cols, ask_key, None)
        if ask is None:
            continue
        if lo <= ask <= hi:
            return i, t, cols
    return None, None, cols

# ─── Extract features at a tick ───────────────────────────────────
def extract_features(tick, cols, market):
    return {
        "cd":           safe(tick, cols, "cd"),
        "dn_ask":       safe(tick, cols, "dn_ask"),
        "up_ask":       safe(tick, cols, "up_ask"),
        "dn_bid":       safe(tick, cols, "dn_bid"),
        "up_bid":       safe(tick, cols, "up_bid"),
        "bn_delta_pct": safe(tick, cols, "bn_delta_pct"),
        "dn_delta_3s":  safe(tick, cols, "dn_delta_3s"),
        "dn_delta_10s": safe(tick, cols, "dn_delta_10s"),
        "up_delta_3s":  safe(tick, cols, "up_delta_3s"),
        "up_delta_10s": safe(tick, cols, "up_delta_10s"),
        "dn_sr":        safe(tick, cols, "dn_sr"),
        "up_sr":        safe(tick, cols, "up_sr"),
        "dn_depth":     safe(tick, cols, "dn_depth"),
        "up_depth":     safe(tick, cols, "up_depth"),
        "dn_zc":        safe(tick, cols, "dn_zc"),
        "up_zc":        safe(tick, cols, "up_zc"),
        "crowd_side":   safe(tick, cols, "crowd_side", ""),
        "conv":         safe(tick, cols, "crowd_conviction"),
        "cl_delta":     safe(tick, cols, "cl_delta"),
        "cl_age":       safe(tick, cols, "cl_age"),
        "ask_sum":      safe(tick, cols, "ask_sum"),
        "bn_spread":    safe(tick, cols, "bn_spread"),
        "winner":       market.get("winner"),
        "slug":         market.get("slug"),
    }

# ═══════════════════════════════════════════════════════════════════
def run_analysis(markets, side, price_lo, price_hi):
    """Full signal analysis for buying `side` when its ask is in [lo, hi]."""
    print(f"\n{'═'*132}")
    print(f"  ANALYSIS: Buy {side} when {side} ask ∈ [${price_lo:.2f}, ${price_hi:.2f}]")
    print(f"{'═'*132}")

    # Collect fires
    fires = []
    for m in markets:
        idx, tick, cols = find_first_tick_in_range(m, side, price_lo, price_hi)
        if tick is None:
            continue
        ask_key = "dn_ask" if side == "DN" else "up_ask"
        ask = safe(tick, cols, ask_key)
        pnl = compute_pnl(m["winner"], ask, DEFAULT_SH, side)
        feats = extract_features(tick, cols, m)
        feats["pnl"] = pnl
        feats["entry_price"] = ask
        fires.append(feats)

    if not fires:
        print(f"  ⚠️ No markets had {side} ask in [${price_lo}, ${price_hi}]")
        return

    # ── Baseline ──
    print(f"\n  ── BASELINE: blind buy {side} at first tick in range ──")
    print(HEADER)
    print(SEP)
    pnls = [f["pnl"] for f in fires]
    print(fmt_row(f"BLIND {side} @ ${price_lo}-${price_hi}", pnls))
    
    # ── Signal A: cd buckets ──
    print(f"\n  ── SIGNAL A: cd (time remaining at entry) ──")
    print(HEADER)
    print(SEP)
    cd_cuts = [
        ("cd ≥ 240",   lambda f: f["cd"] >= 240),
        ("cd 180–239", lambda f: 180 <= f["cd"] < 240),
        ("cd 120–179", lambda f: 120 <= f["cd"] < 180),
        ("cd 60–119",  lambda f: 60  <= f["cd"] < 120),
        ("cd < 60",    lambda f: f["cd"] < 60),
    ]
    for label, pred in cd_cuts:
        sub = [f["pnl"] for f in fires if pred(f)]
        if len(sub) >= 3:
            print(fmt_row(label, sub))

    # ── Signal B: BN momentum ──
    print(f"\n  ── SIGNAL B: Binance momentum (bn_delta_pct) ──")
    if side == "DN":
        print("  Logic: bn < 0 = BTC dropping → favors DN")
    else:
        print("  Logic: bn > 0 = BTC rising → favors UP")
    print(HEADER)
    print(SEP)
    bn_cuts = [
        ("bn < -0.10",       lambda f: f["bn_delta_pct"] < -0.10),
        ("bn −0.10 to −0.02",lambda f: -0.10 <= f["bn_delta_pct"] < -0.02),
        ("bn −0.02 to +0.02",lambda f: -0.02 <= f["bn_delta_pct"] <= 0.02),
        ("bn +0.02 to +0.10",lambda f: 0.02 < f["bn_delta_pct"] <= 0.10),
        ("bn > +0.10",       lambda f: f["bn_delta_pct"] > 0.10),
    ]
    for label, pred in bn_cuts:
        sub = [f["pnl"] for f in fires if pred(f)]
        if len(sub) >= 3:
            print(fmt_row(label, sub))

    # ── Signal C: Δ10s momentum ──
    d10_key = "dn_delta_10s" if side == "DN" else "up_delta_10s"
    print(f"\n  ── SIGNAL C: {side} 10s momentum ({d10_key}) ──")
    print(f"  Logic: positive Δ10s = {side} ask rising = {side} gaining strength")
    print(HEADER)
    print(SEP)
    d10_cuts = [
        (f"Δ10s > +0.05",     lambda f: f[d10_key] > 0.05),
        (f"Δ10s +0.02 to +0.05", lambda f: 0.02 < f[d10_key] <= 0.05),
        (f"Δ10s 0 to +0.02",  lambda f: 0 <= f[d10_key] <= 0.02),
        (f"Δ10s −0.05 to 0",  lambda f: -0.05 <= f[d10_key] < 0),
        (f"Δ10s < −0.05",     lambda f: f[d10_key] < -0.05),
    ]
    for label, pred in d10_cuts:
        sub = [f["pnl"] for f in fires if pred(f)]
        if len(sub) >= 3:
            print(fmt_row(label, sub))

    # ── Signal D: CL delta (regime gate) ──
    print(f"\n  ── SIGNAL D: CL delta (centerline alignment) ──")
    print(f"  Logic: cl > 0 = Binance confirming crowd direction")
    print(HEADER)
    print(SEP)
    cl_cuts = [
        ("cl > +0.03",        lambda f: f["cl_delta"] > 0.03),
        ("cl +0.01 to +0.03", lambda f: 0.01 < f["cl_delta"] <= 0.03),
        ("cl −0.01 to +0.01", lambda f: -0.01 <= f["cl_delta"] <= 0.01),
        ("cl < −0.01",        lambda f: f["cl_delta"] < -0.01),
    ]
    for label, pred in cl_cuts:
        sub = [f["pnl"] for f in fires if pred(f)]
        if len(sub) >= 3:
            print(fmt_row(label, sub))

    # ── Signal E: conv (crowd conviction) ──
    print(f"\n  ── SIGNAL E: crowd conviction ──")
    print(f"  Logic: high conv = crowd very confident → contrarian bet riskier")
    print(HEADER)
    print(SEP)
    conv_cuts = [
        ("conv ≥ 0.70",       lambda f: f["conv"] >= 0.70),
        ("conv 0.60–0.69",    lambda f: 0.60 <= f["conv"] < 0.70),
        ("conv 0.50–0.59",    lambda f: 0.50 <= f["conv"] < 0.60),
        ("conv < 0.50",       lambda f: f["conv"] < 0.50),
    ]
    for label, pred in conv_cuts:
        sub = [f["pnl"] for f in fires if pred(f)]
        if len(sub) >= 3:
            print(fmt_row(label, sub))

    # ── Signal F: SR (depth strength ratio) ──
    sr_key = "dn_sr" if side == "DN" else "up_sr"
    opp_sr_key = "up_sr" if side == "DN" else "dn_sr"
    print(f"\n  ── SIGNAL F: {side} depth strength ratio ({sr_key}) ──")
    print(HEADER)
    print(SEP)
    sr_cuts = [
        (f"{side}_sr ≥ 3.0",    lambda f: f[sr_key] >= 3.0),
        (f"{side}_sr 1.0–3.0",  lambda f: 1.0 <= f[sr_key] < 3.0),
        (f"{side}_sr 0.5–1.0",  lambda f: 0.5 <= f[sr_key] < 1.0),
        (f"{side}_sr < 0.5",    lambda f: f[sr_key] < 0.5),
    ]
    for label, pred in sr_cuts:
        sub = [f["pnl"] for f in fires if pred(f)]
        if len(sub) >= 3:
            print(fmt_row(label, sub))

    # ═══════════════════════════════════════════════════════════════
    # GRID SEARCH: best multi-feature combos
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  ── GRID SEARCH: best (cd × bn × Δ10s × cl × conv) combos ──")
    print(f"  Minimum {MIN_SAMPLE} fires per combo. Ranked by Sharpe × √n.")
    print(HEADER)
    print(SEP)

    # Define filter dimensions
    cd_filters = [
        ("cd≥120",  lambda f: f["cd"] >= 120),
        ("cd≥180",  lambda f: f["cd"] >= 180),
        ("cd≥240",  lambda f: f["cd"] >= 240),
        ("cd<180",  lambda f: f["cd"] < 180),
        ("cd_any",  lambda f: True),
    ]
    
    # BN filters: for DN, negative bn = with-trend; for UP, positive = with-trend
    if side == "DN":
        bn_filters = [
            ("bn<0",     lambda f: f["bn_delta_pct"] < 0),
            ("bn<-0.02", lambda f: f["bn_delta_pct"] < -0.02),
            ("bn>0",     lambda f: f["bn_delta_pct"] > 0),
            ("bn_any",   lambda f: True),
        ]
    else:
        bn_filters = [
            ("bn>0",     lambda f: f["bn_delta_pct"] > 0),
            ("bn>+0.02", lambda f: f["bn_delta_pct"] > 0.02),
            ("bn<0",     lambda f: f["bn_delta_pct"] < 0),
            ("bn_any",   lambda f: True),
        ]

    d10_filters = [
        ("Δ10s>0",    lambda f: f[d10_key] > 0),
        ("Δ10s>+0.02",lambda f: f[d10_key] > 0.02),
        ("Δ10s<0",    lambda f: f[d10_key] < 0),
        ("Δ10s_any",  lambda f: True),
    ]

    cl_filters = [
        ("cl>0",      lambda f: f["cl_delta"] > 0),
        ("cl>+0.01",  lambda f: f["cl_delta"] > 0.01),
        ("cl<0",      lambda f: f["cl_delta"] < 0),
        ("cl_any",    lambda f: True),
    ]

    conv_filters = [
        ("cv<0.60",   lambda f: f["conv"] < 0.60),
        ("cv<0.65",   lambda f: f["conv"] < 0.65),
        ("cv≥0.60",   lambda f: f["conv"] >= 0.60),
        ("cv_any",    lambda f: True),
    ]

    results = []
    for cd_label, cd_pred in cd_filters:
        for bn_label, bn_pred in bn_filters:
            for d10_label, d10_pred in d10_filters:
                for cl_label, cl_pred in cl_filters:
                    for cv_label, cv_pred in conv_filters:
                        # Skip if ALL are _any (that's baseline)
                        n_any = sum(1 for l in [cd_label, bn_label, d10_label,
                                                cl_label, cv_label] if l.endswith("_any"))
                        if n_any == 5:
                            continue
                        # Skip if too many non-any filters (overfitting risk)
                        n_active = 5 - n_any
                        if n_active > 3:
                            continue  # max 3 active filters
                        
                        sub = [f["pnl"] for f in fires
                               if cd_pred(f) and bn_pred(f) and d10_pred(f)
                               and cl_pred(f) and cv_pred(f)]
                        if len(sub) < MIN_SAMPLE:
                            continue
                        s = stats(sub)
                        if s["mean"] <= 0:
                            continue  # only show EV+ combos
                        ci = bootstrap_ci(sub)
                        score = s["sharpe"] * math.sqrt(s["n"])
                        
                        parts = [l for l in [cd_label, bn_label, d10_label,
                                            cl_label, cv_label] if not l.endswith("_any")]
                        label = " + ".join(parts)
                        results.append({
                            "label": label, "pnls": sub, "stats": s,
                            "ci": ci, "score": score,
                        })

    results.sort(key=lambda r: -r["score"])
    
    # Deduplicate by keeping highest-scoring per unique n
    seen_n = set()
    shown = 0
    for r in results:
        if shown >= 20:
            break
        print(fmt_row(r["label"], r["pnls"]))
        shown += 1

    # ═══════════════════════════════════════════════════════════════
    # BEST CANDIDATE: show detail
    # ═══════════════════════════════════════════════════════════════
    if results:
        best = results[0]
        bs = best["stats"]
        bci = best["ci"]
        print(f"\n  ── BEST CANDIDATE ──")
        print(f"  Filter:  {best['label']}")
        print(f"  n={bs['n']}  WR={bs['wr']:.1%}  μ=${bs['mean']:+.4f}/fire  "
              f"Sharpe={bs['sharpe']:+.3f}  p={bs['p']:.4f}")
        if bci[0] is not None:
            print(f"  95% CI: [${bci[0]:+.3f}, ${bci[1]:+.3f}]")
            if bci[0] > 0:
                print(f"  ★★★ CI EXCLUDES ZERO — statistically significant EV+")
            else:
                print(f"  ⚠️ CI includes zero — not yet statistically significant")
        print(f"  Breakeven WR at avg entry: {price_lo:.0%}")
        print(f"  Actual WR: {bs['wr']:.1%} ({'ABOVE' if bs['wr'] > price_lo else 'BELOW'} breakeven)")

    # ═══════════════════════════════════════════════════════════════
    # PRICE SUB-RANGES within $0.40-$0.50
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  ── PRICE SUB-RANGES (where exactly in $0.40-$0.50 is best?) ──")
    print(HEADER)
    print(SEP)
    sub_ranges = [
        (0.40, 0.42), (0.42, 0.44), (0.44, 0.46), (0.46, 0.48), (0.48, 0.50),
        (0.40, 0.45), (0.45, 0.50),
    ]
    for lo, hi in sub_ranges:
        sub = [f["pnl"] for f in fires if lo <= f["entry_price"] <= hi]
        if len(sub) >= 3:
            print(fmt_row(f"{side} @ ${lo:.2f}–${hi:.2f}", sub))

    return fires

# ═══════════════════════════════════════════════════════════════════
# EXTENDED RANGES: also test $0.30-$0.40 and $0.50-$0.60
# ═══════════════════════════════════════════════════════════════════
def run_extended_ranges(markets, side):
    """Test multiple price bands to find the optimal entry zone."""
    print(f"\n{'═'*132}")
    print(f"  EXTENDED RANGE SCAN: {side} entry across all price bands")
    print(f"{'═'*132}")
    print(HEADER)
    print(SEP)
    
    ranges = [
        (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
        (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.45),
        (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
    ]
    
    for lo, hi in ranges:
        fires = []
        for m in markets:
            idx, tick, cols = find_first_tick_in_range(m, side, lo, hi)
            if tick is None:
                continue
            ask_key = "dn_ask" if side == "DN" else "up_ask"
            ask = safe(tick, cols, ask_key)
            pnl = compute_pnl(m["winner"], ask, DEFAULT_SH, side)
            fires.append(pnl)
        if len(fires) >= 3:
            print(fmt_row(f"{side} @ ${lo:.2f}–${hi:.2f}", fires))

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "market_history.jsonl"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)
    
    print(f"Loading markets from {path}...")
    markets = load_markets(path)
    print(f"Loaded {len(markets)} markets with tick data.")
    
    # Base rates
    n_dn = sum(1 for m in markets if m["winner"] == "DN")
    n_up = sum(1 for m in markets if m["winner"] == "UP")
    total = n_dn + n_up
    print(f"\nBase rates: DN={n_dn}/{total} ({n_dn/total:.1%})  "
          f"UP={n_up}/{total} ({n_up/total:.1%})")
    print(f"Breakeven WR at $0.45 entry: 45.0%")
    print(f"Blind EV at $0.45 with DN base rate: "
          f"${(n_dn/total) * 0.55 - (n_up/total) * 0.45:+.4f}/share")

    # ── DN analysis ──
    dn_fires = run_analysis(markets, "DN", DN_LO, DN_HI)
    
    # ── UP analysis (symmetric) ──
    up_fires = run_analysis(markets, "UP", UP_LO, UP_HI)
    
    # ── Extended range scan ──
    run_extended_ranges(markets, "DN")
    run_extended_ranges(markets, "UP")

    # ── Summary ──
    print(f"\n{'═'*132}")
    print("  SUMMARY & RECOMMENDATIONS")
    print(f"{'═'*132}")
    print("""
  Decision framework for implementing a mid-price strategy:
  
  1. If BEST CANDIDATE has CI excluding zero (★★★):
     → Implement as SHADOW strategy, collect 30+ more fires to confirm
     → Then promote to LIVE with conservative sizing (2-3 shares)
  
  2. If BEST CANDIDATE has mean > 0 but CI includes zero (★★ or ★):
     → Implement as SHADOW-only, need more data
     → Do NOT go LIVE until CI tightens
  
  3. If NO combo shows mean > 0 with n ≥ 15:
     → The $0.40-$0.50 zone is NOT exploitable with current signals
     → Focus on proven lottery ($0.05-$0.15) and late-conviction strategies
  
  4. Compare DN vs UP results — if one side dominates, that's the
     direction to build the strategy for.
""")

if __name__ == "__main__":
    main()
