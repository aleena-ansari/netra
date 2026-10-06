"""Parse message_sanitized into diagnostic columns (prefix '_', never raw features).

Direction comes from the message text, not the src_ip/dst_ip columns (see config.yaml).
"""
import ipaddress

import numpy as np
import pandas as pd

# 100.64.0.0/10 is deliberately NOT internal; the data suggests it is the external side.
INTERNAL_NETS = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
PROTO_NAMES = {"1": "icmp", "6": "tcp", "17": "udp", "47": "gre"}

CISCO_RE = (
    r'^<\d+>(?P<mon>\w{3})\s+(?P<day>\d+)\s+(?P<year>\d{4})\s+(?P<time>[\d:]{8}):\s+'
    r'%ASA-(?P<sev>\d)-(?P<code>\d+):\s+(?P<verb>\w+)\s+(?P<proto>\w+)\s+'
    r'src\s+(?P<src_zone>[\w.-]+):(?P<src_ip>[\d.]+)(?:/(?P<src_port>\d+))?\s+'
    r'dst\s+(?P<dst_zone>[\w.-]+):(?P<dst_ip>[\d.]+)(?:/(?P<dst_port>\d+))?'
    r'(?:.*?by\s+access-group\s+"(?P<acl>[^"]+)")?'
)


def is_internal(ips: pd.Series) -> pd.Series:
    """True where the IP is in RFC1918 space. Looks up unique IPs only (fast)."""
    lut = {}
    for ip in ips.dropna().unique():
        try:
            a = ipaddress.ip_address(ip)
            lut[ip] = any(a in n for n in INTERNAL_NETS)
        except ValueError:
            lut[ip] = False
    return ips.map(lut).fillna(False).astype(bool)


def parse_aws(msgs: pd.Series) -> pd.DataFrame:
    """AWS VPC flow log, default 14-field format."""
    f = msgs.str.strip().str.split(expand=True)
    if f.shape[1] < 14:
        raise ValueError(f"expected 14 fields, got {f.shape[1]}")
    out = pd.DataFrame(index=msgs.index)
    out["_src_ip"], out["_dst_ip"] = f[3], f[4]
    out["_sport"] = pd.to_numeric(f[5], errors="coerce")
    out["_dport"] = pd.to_numeric(f[6], errors="coerce")
    out["_proto"] = f[7].map(PROTO_NAMES).fillna("other")
    out["_pkts"] = pd.to_numeric(f[8], errors="coerce")
    out["_bytes"] = pd.to_numeric(f[9], errors="coerce")
    out["_flow_start"] = pd.to_datetime(pd.to_numeric(f[10], errors="coerce"), unit="s", utc=True)
    out["_src_zone"] = pd.NA
    out["_dst_zone"] = pd.NA
    out["_acl"] = pd.NA
    return out


def parse_cisco(msgs: pd.Series) -> pd.DataFrame:
    """Cisco ASA 106023 deny lines. Ports are optional (ICMP has none)."""
    c = msgs.str.extract(CISCO_RE)
    out = pd.DataFrame(index=msgs.index)
    out["_src_ip"], out["_dst_ip"] = c["src_ip"], c["dst_ip"]
    out["_sport"] = pd.to_numeric(c["src_port"], errors="coerce")
    out["_dport"] = pd.to_numeric(c["dst_port"], errors="coerce")
    out["_proto"] = c["proto"].str.lower()
    out["_pkts"] = np.nan
    out["_bytes"] = np.nan
    out["_flow_start"] = pd.NaT
    out["_src_zone"], out["_dst_zone"], out["_acl"] = c["src_zone"], c["dst_zone"], c["acl"]
    return out


def parse_messages(df: pd.DataFrame, source_col="stream_name", msg_col="message_sanitized") -> pd.DataFrame:
    """Return diagnostic '_' columns aligned to df.index. Join with df.join(...)."""
    parts = []
    for name, fn in (("aws_vpc_flow_log", parse_aws), ("cisco_asa", parse_cisco)):
        m = df[source_col] == name
        if m.any():
            parts.append(fn(df.loc[m, msg_col]))
    p = pd.concat(parts).reindex(df.index)

    p["_src_internal"] = is_internal(p["_src_ip"])
    p["_dst_internal"] = is_internal(p["_dst_ip"])
    p["_dir"] = np.select(
       [~p["_src_internal"] & p["_dst_internal"],
        p["_src_internal"] & ~p["_dst_internal"],
        p["_src_internal"] & p["_dst_internal"]],
       ["inbound", "outbound", "internal"],
       default="external",
    )

    # Cisco: zones are more reliable than IP ranges (some 'outside' IPs are RFC1918)
    sz = p["_src_zone"].fillna("").astype(str)
    dz = p["_dst_zone"].fillna("").astype(str)
    has_zone = sz != ""
    p.loc[has_zone & (sz == "outside"), "_dir"] = "inbound"
    p.loc[has_zone & (sz != "outside") & (dz == "outside"), "_dir"] = "outbound"
    p.loc[has_zone & (sz != "outside") & (dz != "outside"), "_dir"] = "internal"
    return p