#!/usr/bin/env python3
"""Research the delta-neutral funding-carry edge on crypto perpetuals (OKX).

The trade: hold spot long + perpetual short in equal size (delta-neutral), and
collect the funding that leveraged longs pay to shorts. This has a STRUCTURAL
basis (funding is persistently positive in bull phases because retail crowds
the long side with leverage) -- unlike a mined technical indicator.

This script pulls real OKX funding-rate history and answers: is the carry, net
of realistic costs, positive and large enough to matter?

Public data, no API key. Run: python research/crypto_funding.py
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.request

# OKX pays funding every 8h -> 3x/day -> ~1095 periods/year.
PERIODS_PER_YEAR = 3 * 365
INSTRUMENTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
               "XRP-USDT-SWAP", "DOGE-USDT-SWAP"]
# Round-trip taker cost to OPEN and later CLOSE both legs (spot+perp), one-time.
ROUND_TRIP_COST = 0.004  # ~0.4% all-in (0.1% * 4 legs); amortized over holding.

_CTX = ssl.create_default_context()
try:  # trust the agent proxy's CA if present (dev container)
    _CTX.load_verify_locations("/root/.ccr/ca-bundle.crt")
except Exception:
    pass


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
        return json.loads(r.read().decode())


def funding_history(inst: str, pages: int = 20) -> list[float]:
    """Return funding rates (most-recent-first), paginating back in time."""
    rates: list[float] = []
    after = ""
    for _ in range(pages):
        url = (f"https://www.okx.com/api/v5/public/funding-rate-history?"
               f"instId={inst}&limit=100")
        if after:
            url += f"&after={after}"
        data = _get(url).get("data", [])
        if not data:
            break
        for row in data:
            rates.append(float(row["realizedRate"]))
        after = data[-1]["fundingTime"]
        time.sleep(0.15)
    return rates


def analyse(inst: str) -> dict:
    rates = funding_history(inst)
    if not rates:
        return {"inst": inst, "n": 0}
    n = len(rates)
    mean = sum(rates) / n
    pos = sum(1 for r in rates if r > 0) / n
    recent = rates[:270]  # ~90 days (3/day)
    mean_recent = sum(recent) / len(recent)
    ann = mean * PERIODS_PER_YEAR
    ann_recent = mean_recent * PERIODS_PER_YEAR
    # Net annual carry assumes you hold ~1 quarter (4 round trips/yr of cost).
    net_ann = ann - ROUND_TRIP_COST * 4
    return {
        "inst": inst, "n": n, "pos_pct": pos,
        "ann_all": ann, "ann_recent": ann_recent, "net_ann": net_ann,
    }


def main() -> None:
    print(f"OKX funding-carry research  (periods/yr={PERIODS_PER_YEAR}, "
          f"round-trip cost {ROUND_TRIP_COST:.1%})\n")
    print(f"{'instrument':<16}{'periods':>8}{'%pos':>7}"
          f"{'ann(all)':>11}{'ann(90d)':>11}{'net ann':>10}")
    print("-" * 63)
    for inst in INSTRUMENTS:
        try:
            r = analyse(inst)
        except Exception as exc:  # noqa: BLE001
            print(f"{inst:<16} ERROR: {exc}")
            continue
        if r.get("n", 0) == 0:
            print(f"{inst:<16} no data")
            continue
        print(f"{r['inst']:<16}{r['n']:>8}{r['pos_pct']*100:>6.0f}%"
              f"{r['ann_all']*100:>10.1f}%{r['ann_recent']*100:>10.1f}%"
              f"{r['net_ann']*100:>9.1f}%")
    print("-" * 63)
    print("ann = annualized carry from collecting funding (delta-neutral).")
    print("net ann subtracts ~4 round-trips/yr of trading cost.")
    print("Positive net ann = the carry beats costs. Compare vs a bank/T-bill")
    print("rate: the edge must clear the RISK (exchange, liquidation, tails).")


if __name__ == "__main__":
    main()
