#!/usr/bin/env python3
"""Train + out-of-sample evaluate the gold meta-labeling model.

Usage:
    python train_gold_ml.py [--symbol XAUUSD] [--tf H1] [--bars 6000]
                            [--threshold 0.55] [--cost-atr 0.1]

Pulls history from the logged-in MT5 terminal, builds primary breakout signals,
labels them with the triple-barrier method, trains a meta-label classifier on
the first 70% and evaluates on the unseen last 30%, then compares taking ALL
primary signals vs taking only ML-approved ones. Saves the model to
models/gold_meta.joblib for later use.

Honest reading: the ML layer earns its place ONLY if it raises out-of-sample
expectancy. If gated OOS expectancy is not clearly better than ungated, the
model adds nothing and should not be used.
"""
from __future__ import annotations

import sys

import numpy as np

from trade_bot.ml import MetaLabeler
from trade_bot.ml.pipeline import build_dataset
from trade_bot.logging_setup import setup_logging
from trade_bot.mt5_client import MT5Client, timeframe_const

DEFAULTS = {"symbol": "XAUUSD", "tf": "H1", "bars": 6000,
            "threshold": 0.55, "cost_atr": 0.1,
            "sl_atr": 1.5, "tp_atr": 2.0, "horizon": 24}


def _arg(args: list[str], name: str, cast, default):
    flag = f"--{name}"
    if flag in args:
        i = args.index(flag)
        return cast(args[i + 1])
    return default


def _payoff(labels, sl_atr, tp_atr, cost_atr) -> np.ndarray:
    return np.where(labels == 1, tp_atr, -sl_atr) - cost_atr


def _expectancy(labels, sl_atr, tp_atr, cost_atr) -> dict:
    """Trade stats in ATR (R) units, with a t-stat for significance."""
    if len(labels) == 0:
        return {"n": 0, "win_rate": 0.0, "expectancy_R": 0.0, "t": 0.0}
    payoff = _payoff(labels, sl_atr, tp_atr, cost_atr)
    mean = float(payoff.mean())
    std = float(payoff.std(ddof=1)) if len(payoff) > 1 else 0.0
    t = mean / (std / np.sqrt(len(payoff))) if std > 0 else 0.0
    return {
        "n": int(len(labels)),
        "win_rate": float((labels == 1).mean()),
        "expectancy_R": mean,
        "t": float(t),
    }


def _cost_sweep(labels, sl_atr, tp_atr) -> str:
    cells = []
    for c in (0.0, 0.1, 0.2, 0.3):
        e = _expectancy(labels, sl_atr, tp_atr, c)["expectancy_R"]
        cells.append(f"{c:>4}:{e:+.3f}")
    return "  ".join(cells)


def _robustness_breakdown(meta, sl_atr, tp_atr, cost_atr) -> None:
    """The decisive test: is the edge stable across TIME and SIDE, or is it
    just riding the recent gold trend (one period, long-only)?"""
    labels = meta["label"].to_numpy()
    sides = meta["side"].to_numpy()
    print("-" * 60)
    print(f"  ROBUSTNESS (cost {cost_atr} R) -- a real edge is positive across")
    print("  ALL time folds AND on both sides; one good fold = regime luck:")
    folds = np.array_split(np.arange(len(labels)), 5)
    print(f"    {'fold':<10}{'n':>6}{'win%':>8}{'exp R':>10}")
    for k, idx in enumerate(folds, 1):
        e = _expectancy(labels[idx], sl_atr, tp_atr, cost_atr)
        print(f"    {('T'+str(k)):<10}{e['n']:>6}{e['win_rate']*100:>7.0f}%{e['expectancy_R']:>10.3f}")
    for name, mask in (("long", sides == 1), ("short", sides == -1)):
        e = _expectancy(labels[mask], sl_atr, tp_atr, cost_atr)
        print(f"    {name:<10}{e['n']:>6}{e['win_rate']*100:>7.0f}%{e['expectancy_R']:>10.3f}")


def main() -> int:
    args = sys.argv[1:]
    symbol = _arg(args, "symbol", str, DEFAULTS["symbol"])
    tf = _arg(args, "tf", str, DEFAULTS["tf"])
    bars = _arg(args, "bars", int, DEFAULTS["bars"])
    threshold = _arg(args, "threshold", float, DEFAULTS["threshold"])
    cost_atr = _arg(args, "cost-atr", float, DEFAULTS["cost_atr"])
    sl_atr, tp_atr, horizon = DEFAULTS["sl_atr"], DEFAULTS["tp_atr"], DEFAULTS["horizon"]

    setup_logging("INFO", None)
    print(f"Loading {bars} {tf} bars of {symbol} from MT5...")
    client = MT5Client()
    client.connect()
    client.ensure_symbols([symbol])
    df = client.rates(symbol, timeframe_const(tf), bars)
    client.shutdown()

    X, y, meta = build_dataset(df, sl_atr=sl_atr, tp_atr=tp_atr, horizon=horizon)
    if len(X) < 200:
        print(f"Only {len(X)} labeled signals -- not enough to train. "
              f"Try more --bars or a faster timeframe.")
        return 1

    # Time-ordered split (NEVER shuffle financial data).
    split = int(len(X) * 0.7)
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xte, yte = X.iloc[split:], y.iloc[split:]
    lab_tr = meta["label"].to_numpy()[:split]
    lab_te = meta["label"].to_numpy()[split:]

    clf = MetaLabeler()
    clf.fit(Xtr, ytr)
    tr_m, te_m = clf.evaluate(Xtr, ytr), clf.evaluate(Xte, yte)

    proba_te = clf.predict_proba(Xte)
    gated = proba_te >= threshold
    ungated_stats = _expectancy(lab_te, sl_atr, tp_atr, cost_atr)
    gated_stats = _expectancy(lab_te[gated], sl_atr, tp_atr, cost_atr)

    clf.save("models/gold_meta.joblib")

    print("\n" + "=" * 60)
    print(f"  GOLD META-LABEL MODEL  ({symbol} {tf})")
    print("=" * 60)
    print(f"  Signals: {len(X)} total | train {len(Xtr)} | test(OOS) {len(Xte)}")
    print(f"  Primary-signal base win rate (train): {tr_m['base_rate']:.1%}")
    print(f"  Classifier AUC  : train {tr_m['auc']:.3f} | OOS {te_m['auc']:.3f}")
    print(f"  Classifier acc. : train {tr_m['accuracy']:.3f} | OOS {te_m['accuracy']:.3f}")
    print("-" * 60)
    # The RAW primary signal is the real subject of interest: does the breakout
    # rule itself have an edge, in both halves and across realistic costs?
    is_raw = _expectancy(lab_tr, sl_atr, tp_atr, cost_atr)
    oos_raw = _expectancy(lab_te, sl_atr, tp_atr, cost_atr)
    print(f"  RAW PRIMARY SIGNAL (no ML), R = ATR units, cost {cost_atr} R:")
    print(f"    {'':<16}{'in-sample':>14}{'out-of-sample':>16}")
    print(f"    {'trades':<16}{is_raw['n']:>14}{oos_raw['n']:>16}")
    print(f"    {'win rate':<16}{is_raw['win_rate']*100:>13.1f}%{oos_raw['win_rate']*100:>15.1f}%")
    print(f"    {'expectancy R':<16}{is_raw['expectancy_R']:>14.3f}{oos_raw['expectancy_R']:>16.3f}")
    print(f"    {'t-stat':<16}{is_raw['t']:>14.2f}{oos_raw['t']:>16.2f}")
    print(f"    cost sweep IS : {_cost_sweep(lab_tr, sl_atr, tp_atr)}")
    print(f"    cost sweep OOS: {_cost_sweep(lab_te, sl_atr, tp_atr)}")
    _robustness_breakdown(meta, sl_atr, tp_atr, cost_atr)
    print("-" * 60)
    print(f"  ML GATING vs RAW (OOS, cost {cost_atr} R):")
    print(f"    take ALL : n={ungated_stats['n']:<4} exp={ungated_stats['expectancy_R']:+.3f} R")
    print(f"    ML-gated : n={gated_stats['n']:<4} exp={gated_stats['expectancy_R']:+.3f} R")
    print("=" * 60)
    # Verdicts: ML and the raw signal are judged separately.
    if te_m["auc"] < 0.52 or gated_stats["expectancy_R"] <= ungated_stats["expectancy_R"]:
        print("  ML: no value (OOS AUC ~0.5 or gating doesn't beat raw). Drop it.")
    else:
        print("  ML: gating improves OOS expectancy -- worth keeping.")
    if oos_raw["expectancy_R"] > 0 and oos_raw["t"] >= 2.0:
        print("  RAW SIGNAL: positive AND statistically significant (t>=2). Strong.")
    elif oos_raw["expectancy_R"] > 0:
        print("  RAW SIGNAL: positive but NOT yet significant (t<2). Promising;")
        print("  needs more trades/data + realistic costs before trusting.")
    else:
        print("  RAW SIGNAL: no positive out-of-sample edge.")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
