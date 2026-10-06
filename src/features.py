"""Static fields + past-only rolling features. Keys (m_src_ip, m_dst_ip) group rows
but are NEVER returned as features. 'Past' follows feature_policy.tie_rule:
  ordered_by_event_id : rows earlier in (timestamp, event_id) order
  strict_past         : rows with a strictly earlier timestamp
Windows keep rows with timestamp >= t - w. No labels are used anywhere."""
from collections import deque, Counter, defaultdict
import numpy as np
import pandas as pd
from . import parse_messages as pm

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

def rolling_past(ts, key, port, dst, windows, tie_rule):
    """Inputs are equal-length Series already sorted by (timestamp, event_id)."""
    t_arr = (ts - EPOCH).dt.total_seconds().to_numpy()
    keys, ports, dsts = key.to_numpy(), port.to_numpy(dtype=float), dst.to_numpy()
    n = len(t_arr)
    out = {f"{nm}_{w}s": np.zeros(n, dtype="int32") for w in windows for nm in ("cnt", "nports", "ndst")}
    state = [defaultdict(lambda: [deque(), Counter(), Counter()]) for _ in windows]

    def add(k, t, p, ds):
        for st in state:
            dq, cp, cd = st[k]
            dq.append((t, p, ds))
            if p == p: cp[p] += 1            # p == p is False for NaN
            cd[ds] += 1

    def evict(s, t, w):
        dq, cp, cd = s
        while dq and dq[0][0] < t - w:
            _, p, ds = dq.popleft()
            if p == p:
                cp[p] -= 1
                if cp[p] == 0: del cp[p]
            cd[ds] -= 1
            if cd[ds] == 0: del cd[ds]

    strict, pending, pend_t = tie_rule == "strict_past", [], None
    for i in range(n):
        t = t_arr[i]
        if strict and pending and pend_t != t:     # same-second rows become visible only later
            for r in pending: add(*r)
            pending = []
        for j, w in enumerate(windows):
            s = state[j][keys[i]]
            evict(s, t, w)
            out[f"cnt_{w}s"][i], out[f"nports_{w}s"][i], out[f"ndst_{w}s"][i] = len(s[0]), len(s[1]), len(s[2])
        row = (keys[i], t, ports[i], dsts[i])
        if strict:
            pending.append(row); pend_t = t
        else:
            add(*row)                              # features computed BEFORE adding the current row
    return pd.DataFrame(out)

def static_features(g, stream):
    if stream == "aws_vpc_flow_log":
        p = pm.parse_aws(g["message_sanitized"]).rename(columns={"srcport": "src_port", "dstport": "dst_port"})
        X = p[["src_port", "dst_port", "protocol", "packets", "bytes", "duration"]].copy()
        X["bytes_per_packet"] = X["bytes"] / X["packets"].clip(lower=1)
    else:
        p = pm.parse_cisco(g["message_sanitized"])
        X = p[["proto", "src_zone", "dst_zone", "acl", "src_port", "dst_port",
               "icmp_type", "icmp_code"]].copy()
    X["dst_port_wellknown"] = (X["dst_port"] < 1024).astype("int8")
    return X, p[["m_src_ip", "m_dst_ip"]]

def build_features(d, cfg, tie_rule=None):
    """Returns {stream: feature DataFrame}, rows sorted by (timestamp, event_id), index = d's index."""
    pol = cfg["feature_policy"]
    tie_rule = tie_rule or pol["tie_rule"]
    res = {}
    for stream, g in d.groupby("stream_name"):
        g = g.sort_values(["timestamp", "event_id"])
        X, keys = static_features(g, stream)
        R = rolling_past(g["timestamp"], keys["m_src_ip"], X["dst_port"], keys["m_dst_ip"],
                         pol["window_seconds"], tie_rule)
        R.index = g.index
        res[stream] = pd.concat([X, R], axis=1)
    return res
