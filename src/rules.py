"""Sigma-style rules (simplified). A rule has id, title, logsource.stream, detection.selection,
level, tags. In a selection: scalar = equality, list = any-of, 'field|gte/gt/lte/lt/ne' = comparison.
All keys are ANDed. Fields are parsed fields and past-only rolling features. Rules never see labels."""
import operator
from pathlib import Path
import pandas as pd
import yaml

OPS = {"gte": operator.ge, "gt": operator.gt, "lte": operator.le, "lt": operator.lt, "ne": operator.ne}

def load_rules(rules_dir):
    rules = []
    for p in sorted(Path(rules_dir).glob("*.yaml")):
        rules += yaml.safe_load(p.read_text(encoding="utf-8")) or []
    return rules

def _match(frame, selection):
    m = pd.Series(True, index=frame.index)
    for key, val in selection.items():
        field, _, op = key.partition("|")
        col = frame[field]                       # unknown field -> KeyError (loud on purpose)
        if op:
            m &= OPS[op](col, val)
        else:
            m &= col.isin(val) if isinstance(val, list) else (col == val)
    return m

def apply_rules(frame, stream, rules):
    """Returns a bool DataFrame (rows x rule ids) for the rules of this stream."""
    out = {r["id"]: _match(frame, r["detection"]["selection"])
           for r in rules if r["logsource"]["stream"] == stream}
    return pd.DataFrame(out, index=frame.index)
