"""Local JevBench scorer, following method v1.6 (O1S) as published. Approximate where the method is not public.

Intelligence (METHOD-v1.5 section 3, v1.5 equal-types addendum, v1.6 O1S note on the board):
  choice  CC = 100 (acc - mean 1/n) / (1 - mean 1/n), top answer
  noul    P(yes) >= 0.80 is yes, <= 0.20 is no, in between is an abstention = wrong; CC = 100 (acc - 0.5) / 0.5
  score   prediction = expected level; nMAE = |pred - gold| / (K-1); CC = 100 (1 - mean nMAE / mean nMAE_mid), nMAE_mid
          = error of always predicting mid-scale (K-1)/2 (O1S); a cell whose golds all sit at mid-scale uses the
          random-level baseline
  per split x type: tiers weighted easy .1 / standard .2 / judge .3 / hard .4 (renormalised over present tiers), then
  clipped at 0 (O1S); types 1/3 each; I = .5 open + .5 sealed (local items are open only, so I = I_open);
  overfit penalty not applied (needs sealed).
  An answer over 10 s (decision_limit) counts as wrong (v1.7).
Calibration (approximation): per type, pooled over splits and tiers.
  choice  100 (1 - ECE_top - 0.5 TVD), TVD = mean total variation to gold_probs on items that have them. The form
          100 - 100 ECE - 50 TVD reproduces v1.4.2's published calibration of gpt-6-luna exactly (89.71).
  noul    100 (1 - ECE of P(yes)) (10 equal-width bins of P(yes) against the yes rate)
  score   100 (1 - ECE_top - 0.5 nRPS) (assumed by analogy; the exact combination is not published)
  invalid answers: wrong for intelligence, confidence 1 and wrong for ECE.
Capability = (I + C) / 2. Cost: mean input tokens per decision and USD per 1,000 at Gemma-4-31B's USD 0.09/M.
"""
import json
import math
from collections import defaultdict

TIER_W = {"easy": 0.10, "standard": 0.20, "judge": 0.30, "hard": 0.40}
TYPES = ("choice", "noul", "score")
DECISION_LIMIT_S = 10.0
PRICE_IN_PER_M = 0.09
COST_CAP_USD_PER_1000 = 0.06459465517241379
LATENCY_CAP_S = 1.2329566404223442


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def valid_probs(probs, labels):
    if not isinstance(probs, dict):
        return False
    try:
        vals = [float(probs[str(l)]) for l in labels]
    except (KeyError, TypeError, ValueError):
        return False
    return all(0 <= v <= 1 for v in vals) and abs(sum(vals) - 1) <= 0.02


def ece(pairs, bins=10):
    """pairs of (confidence, outcome in {0,1}) -> expected calibration error over equal-width bins."""
    acc = defaultdict(lambda: [0, 0.0, 0.0])
    for c, y in pairs:
        b = min(int(min(max(c, 0.0), 1.0) * bins), bins - 1)
        acc[b][0] += 1
        acc[b][1] += c
        acc[b][2] += y
    n = sum(v[0] for v in acc.values())
    return sum(v[0] / n * abs(v[2] / v[0] - v[1] / v[0]) for v in acc.values()) if n else 0.0


def item_outcome(g, r):
    """-> dict with type, tier, split, correct (choice/noul), nmae / nmae_chance (score), calibration inputs."""
    kind, labels = g["question"]["type"], [str(l) for l in g["labels"]]
    out = {"type": kind, "tier": g["tier"], "split": g.get("split", "open"), "id": g["id"]}
    ok = bool(r) and r.get("ok", True) and not r.get("error") and valid_probs(r.get("probs"), labels)
    late = bool(r) and (r.get("latency_s") or 0) > DECISION_LIMIT_S
    out["valid"], out["late"] = ok, late
    p = [float(r["probs"][l]) for l in labels] if ok else None
    if kind == "choice":
        out["chance"] = 1 / len(labels)
        top = max(range(len(labels)), key=p.__getitem__) if ok else None
        out["correct"] = ok and not late and labels[top] == str(g["expected"])
        out["conf"] = p[top] if ok and not late else 1.0
        out["conf_correct"] = out["correct"]
        if g.get("gold_probs") and ok and not late:
            out["tvd"] = 0.5 * sum(abs(p[i] - float(g["gold_probs"].get(l, 0))) for i, l in enumerate(labels))
    elif kind == "noul":
        py = p[labels.index("yes")] if ok else None
        y = 1 if str(g["expected"]) == "yes" else 0
        dec = None if py is None else ("yes" if py >= 0.80 else "no" if py <= 0.20 else None)
        out["decisive"] = dec is not None
        out["correct"] = ok and not late and dec == str(g["expected"])
        out["py"] = py if ok and not late else (1.0 - y)     # invalid: confident and wrong
        out["y"] = y
    elif kind == "score":
        k = len(labels)
        gold = int(g["expected"])
        mid = (k - 1) / 2
        pred = sum(i * v for i, v in enumerate(p)) if ok and not late else (0 if gold > mid else k - 1)
        out["nmae"] = abs(pred - gold) / (k - 1)
        out["nmae_mid"] = abs(mid - gold) / (k - 1)
        out["nmae_rand"] = sum(abs(l - gold) for l in range(k)) / k / (k - 1)
        if ok and not late:
            top = max(range(k), key=p.__getitem__)
            out["conf"], out["conf_correct"] = p[top], top == gold
            cdf_p = [sum(p[:i + 1]) for i in range(k - 1)]
            cdf_g = [1.0 if i >= gold else 0.0 for i in range(k - 1)]
            out["nrps"] = sum((a - b) ** 2 for a, b in zip(cdf_p, cdf_g)) / (k - 1)
        else:
            out["conf"], out["conf_correct"], out["nrps"] = 1.0, False, 1.0
    return out


def cc_cell(kind, items):
    if kind == "choice":
        acc = sum(o["correct"] for o in items) / len(items)
        c = sum(o["chance"] for o in items) / len(items)
        return 100 * (acc - c) / (1 - c)
    if kind == "noul":
        acc = sum(o["correct"] for o in items) / len(items)
        return 100 * (acc - 0.5) / 0.5
    base = sum(o["nmae_mid"] for o in items) / len(items)
    if base <= 0:
        base = sum(o["nmae_rand"] for o in items) / len(items)
    return 100 * (1 - (sum(o["nmae"] for o in items) / len(items)) / base)


def score(gold_rows, raw_rows):
    raw = {(r.get("task_id") or r.get("id")): r for r in raw_rows}
    outs = [item_outcome(g, raw.get(g["id"])) for g in gold_rows]
    res = {"n": len(outs), "missing": sum(g["id"] not in raw for g in gold_rows)}
    # intelligence
    per = {}
    split_i = {}
    for split in sorted({o["split"] for o in outs}):
        type_cc = {}
        for kind in TYPES:
            cells = {}
            for tier in TIER_W:
                its = [o for o in outs if o["split"] == split and o["type"] == kind and o["tier"] == tier]
                if its:
                    cells[tier] = (cc_cell(kind, its), len(its))
            if cells:
                wsum = sum(TIER_W[t] for t in cells)
                raw_cc = sum(TIER_W[t] * v for t, (v, _) in cells.items()) / wsum
                type_cc[kind] = max(0.0, raw_cc)
                per[f"{split}|{kind}"] = {"cc": type_cc[kind], "cc_unclipped": raw_cc, "tiers": {t: {"cc": round(v, 2), "n": n} for t, (v, n) in cells.items()}}
        split_i[split] = sum(type_cc.values()) / len(type_cc)
    intel = sum(split_i.values()) / len(split_i)
    # calibration
    parts = {}
    ch = [o for o in outs if o["type"] == "choice"]
    if ch:
        e = ece([(o["conf"], 1 if o["conf_correct"] else 0) for o in ch])
        tv = [o["tvd"] for o in ch if "tvd" in o]
        t = sum(tv) / len(tv) if tv else 0.0
        parts["choice"] = {"ece": e, "tvd": t, "n_tvd": len(tv), "score": 100 * (1 - e - 0.5 * t)}
    no = [o for o in outs if o["type"] == "noul"]
    if no:
        e = ece([(o["py"], o["y"]) for o in no])
        parts["noul"] = {"ece": e, "decisive_rate": sum(o["decisive"] for o in no) / len(no), "score": 100 * (1 - e)}
    sc = [o for o in outs if o["type"] == "score"]
    if sc:
        e = ece([(o["conf"], 1 if o["conf_correct"] else 0) for o in sc])
        r = sum(o["nrps"] for o in sc) / len(sc)
        parts["score"] = {"ece": e, "nrps": r, "score": 100 * (1 - e - 0.5 * r)}
    calib = sum(p["score"] for p in parts.values()) / len(parts)
    res.update(intelligence=intel, intelligence_by_split=split_i, per_type_split=per, calibration=calib, calibration_parts=parts,
               capability=(intel + calib) / 2, accuracy={k: sum(o.get("correct", False) for o in outs if o["type"] == k) / max(1, sum(o["type"] == k for o in outs))
                                                          for k in ("choice", "noul")},
               invalid=sum(not o["valid"] for o in outs), over_limit=sum(o["late"] for o in outs))
    # cost and latency
    toks = [((raw.get(g["id"]) or {}).get("usage") or {}).get("input_tokens") for g in gold_rows]
    toks = [t for t in toks if isinstance(t, (int, float))]
    if toks:
        mean_tok = sum(toks) / len(toks)
        res["tokens_per_decision"] = mean_tok
        res["usd_per_1000"] = mean_tok * PRICE_IN_PER_M / 1e6 * 1000
        res["cost_vs_cap"] = res["usd_per_1000"] / COST_CAP_USD_PER_1000
    lat = sorted(r.get("latency_s") for r in raw_rows if isinstance(r.get("latency_s"), (int, float)))
    if lat:
        q = lambda f: lat[min(len(lat) - 1, int(round(f * (len(lat) - 1))))]
        res["latency_p50_s"], res["latency_p95_s"] = q(0.5), q(0.95)
        res["latency_p50_adjusted_s"] = 2 * q(0.5) + 0.15
    return res


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("raw")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = score(load_jsonl(a.gold), load_jsonl(a.raw))
    if a.json:
        print(json.dumps(res, indent=1))
        return
    print(f"n={res['n']} missing={res['missing']} invalid={res['invalid']} over10s={res['over_limit']}")
    print(f"Capability {res['capability']:.2f} = (I {res['intelligence']:.2f} + C {res['calibration']:.2f}) / 2   [local, open items only, C approximate]")
    for k, v in res["per_type_split"].items():
        print(f"  I {k:<14} {v['cc']:6.2f}  " + "  ".join(f"{t} {d['cc']:.1f} (n={d['n']})" for t, d in v["tiers"].items()))
    for k, v in res["calibration_parts"].items():
        print(f"  C {k:<8} {v['score']:6.2f}  " + "  ".join(f"{a}={b:.4f}" for a, b in v.items() if a != "score"))
    print(f"  accuracy choice {res['accuracy']['choice']:.3f} noul(decisive & right) {res['accuracy']['noul']:.3f}")
    if "tokens_per_decision" in res:
        print(f"  tokens/decision {res['tokens_per_decision']:.0f}  USD/1000 {res['usd_per_1000']:.4f}  = {res['cost_vs_cap']:.2f}x cap")
    if "latency_p50_s" in res:
        print(f"  latency p50 {res['latency_p50_s']:.3f}s p95 {res['latency_p95_s']:.3f}s  adjusted p50 {res['latency_p50_adjusted_s']:.3f}s (cap {LATENCY_CAP_S:.3f})")


if __name__ == "__main__":
    main()
