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


def _expectancy(labels, sl_atr, tp_atr, cost_atr) -> dict:
    """Trade stats in ATR (R) units for a set of taken signals."""
    if len(labels) == 0:
        return {"n": 0, "win_rate": 0.0, "expectancy_R": 0.0}
    wins = labels == 1
    payoff = np.where(wins, tp_atr, -sl_atr) - cost_atr
    return {
        "n": int(len(labels)),
        "win_rate": float(wins.mean()),
        "expectancy_R": float(payoff.mean()),
    }


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
    print(f"  OUT-OF-SAMPLE trade economics (R = ATR units, cost {cost_atr} R):")
    print(f"    {'':<22}{'take ALL':>14}{'ML-gated':>14}")
    print(f"    {'trades taken':<22}{ungated_stats['n']:>14}{gated_stats['n']:>14}")
    print(f"    {'win rate':<22}{ungated_stats['win_rate']*100:>13.1f}%{gated_stats['win_rate']*100:>13.1f}%")
    print(f"    {'expectancy (R/trade)':<22}{ungated_stats['expectancy_R']:>14.3f}{gated_stats['expectancy_R']:>14.3f}")
    print("=" * 60)
    better = gated_stats["expectancy_R"] > max(ungated_stats["expectancy_R"], 0)
    if te_m["auc"] < 0.52:
        print("  VERDICT: OOS AUC ~ 0.5 -> the model has no predictive signal.")
        print("  The ML layer is not helping. Do NOT trade this.")
    elif better:
        print("  VERDICT: ML gating improves OOS expectancy AND it is positive.")
        print("  Promising -- but still validate live on demo before real money.")
    else:
        print("  VERDICT: gating does not produce a positive OOS edge.")
        print("  Honest result: no tradable edge found here.")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
