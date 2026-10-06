"""Structured fields from message_sanitized (the raw log is the source of truth
for IPs, ports and direction). '_'-prefixed columns are diagnostics only.
m_src_ip / m_dst_ip are grouping KEYS for rolling features, never model inputs."""
import pandas as pd

AWS_COLS = ["version", "account_id", "interface_id", "srcaddr", "dstaddr",
            "srcport", "dstport", "protocol", "packets", "bytes",
            "start", "end", "action", "log_status"]
AWS_NUM = ["srcport", "dstport", "protocol", "packets", "bytes"]

CISCO_RE = (
    r'^<\d+>(?P<mtime>[A-Z][a-z]{2} +\d+ \d{4} \d{2}:\d{2}:\d{2}): '
    r'%ASA-\d+-(?P<msg_id>\d+): Deny (?P<proto>\w+) '
    r'src (?P<src_zone>[\w-]+):(?P<m_src_ip>[\d.]+)(?:/(?P<src_port>\d+))? '
    r'dst (?P<dst_zone>[\w-]+):(?P<m_dst_ip>[\d.]+)(?:/(?P<dst_port>\d+))?'
    r'(?: \(type (?P<icmp_type>\d+), code (?P<icmp_code>\d+)\))? '
    r'by access-group "(?P<acl>[^"]+)"'
)

def parse_aws(msg: pd.Series) -> pd.DataFrame:
    msg = msg.fillna("")
    n_tok = msg.str.split().str.len()
    p = msg.str.split(expand=True).reindex(columns=range(14))
    p.columns = AWS_COLS
    out = pd.DataFrame(index=msg.index)
    for c in AWS_NUM:
        out[c] = pd.to_numeric(p[c], errors="coerce")
    start = pd.to_numeric(p["start"], errors="coerce")
    end = pd.to_numeric(p["end"], errors="coerce")
    out["duration"] = end - start
    out["m_src_ip"] = p["srcaddr"]
    out["m_dst_ip"] = p["dstaddr"]
    out["_end_time"] = pd.to_datetime(end, unit="s", utc=True)
    out["_account_id"] = p["account_id"]
    out["_interface_id"] = p["interface_id"]
    out["parse_ok"] = (n_tok == 14) & out[AWS_NUM + ["duration"]].notna().all(axis=1)
    return out

def parse_cisco(msg: pd.Series) -> pd.DataFrame:
    m = msg.fillna("").str.strip().str.extract(CISCO_RE)
    out = pd.DataFrame(index=msg.index)
    out["parse_ok"] = m["proto"].notna()
    out["proto"] = m["proto"]
    out["src_zone"] = m["src_zone"]
    out["dst_zone"] = m["dst_zone"]
    out["acl"] = m["acl"]
    out["msg_id"] = m["msg_id"]
    out["m_src_ip"] = m["m_src_ip"]
    out["m_dst_ip"] = m["m_dst_ip"]
    for c in ["src_port", "dst_port", "icmp_type", "icmp_code"]:
        out[c] = pd.to_numeric(m[c], errors="coerce")   # NaN where not applicable
    out["_msg_time"] = pd.to_datetime(m["mtime"], format="%b %d %Y %H:%M:%S", errors="coerce")
    return out
