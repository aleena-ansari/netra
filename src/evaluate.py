"""Evaluation helpers: same-alert-volume comparison for one stream."""
import numpy as np
import pandas as pd


def _row(label, alert_mask, y, P, hours):
    a = int(alert_mask.sum())
    tp = int((alert_mask & y).sum())
    return dict(strategy=label, alerts=a, malicious_found=tp,
                precision=round(tp / a, 3) if a else None,
                recall=round(tp / P, 3) if P else None,
                alerts_per_hour=round(a / hours, 1))


def same_volume_table(split, y, ts, rule_caught, inbound, score, seed):
    """All inputs are Series on one shared index (one stream); y/rule_caught/inbound are bool.
    The IF and the random baseline get exactly as many extra alerts as 'flag inbound' uses."""
    rng = np.random.default_rng(seed)
    tables = {}
    for name in ["val", "test"]:
        m = split == name
        P = int(y[m].sum())
        hours = (ts[m].max() - ts[m].min()).total_seconds() / 3600
        res = m & ~rule_caught                      # rows the rules did not catch
        inb = res & inbound
        k = int(inb.sum())
        top = pd.Series(False, index=y.index)
        top.loc[score[res].nlargest(k).index] = True
        rnd = pd.Series(False, index=y.index)
        rnd.loc[rng.choice(y.index[res.to_numpy()], size=k, replace=False)] = True
        base = rule_caught & m
        t = pd.DataFrame([
            _row("rules only", base, y, P, hours),
            _row("rules + flag inbound", base | inb, y, P, hours),
            _row(f"rules + IF top-{k}", base | top, y, P, hours),
            _row(f"rules + random {k}", base | rnd, y, P, hours)])
        t.attrs.update(events=int(m.sum()), malicious=P, hours=round(hours, 1))
        tables[name] = t
    return tables


def budget_table(split, y, ts, rule_caught, inbound, score, seed, fracs=(0.05, 0.10, 0.20)):
    """IF vs random vs 'flag inbound' at fixed alert budgets (share of rule-residual rows)."""
    rng = np.random.default_rng(seed)
    tables = {}
    for name in ["val", "test"]:
        m = split == name
        P = int(y[m].sum())
        hours = (ts[m].max() - ts[m].min()).total_seconds() / 3600
        res = m & ~rule_caught
        rows = [_row("flag inbound", res & inbound, y, P, hours)]
        for f in fracs:
            k = int(res.sum() * f)
            top = pd.Series(False, index=y.index)
            top.loc[score[res].nlargest(k).index] = True
            rnd = pd.Series(False, index=y.index)
            rnd.loc[rng.choice(y.index[res.to_numpy()], size=k, replace=False)] = True
            rows.append(_row(f"IF top {int(f*100)}% ({k})", top, y, P, hours))
            rows.append(_row(f"random {int(f*100)}% ({k})", rnd, y, P, hours))
        t = pd.DataFrame(rows)
        t.attrs.update(events=int(m.sum()), malicious=P,
                       base_rate=round(P / m.sum(), 3), hours=round(hours, 1))
        tables[name] = t
    return tables