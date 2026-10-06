"""Run from the project root:  python tests\test_xgb_stage.py
The two sample rows are synthetic probes (not real attacks); they only exercise
the high-confidence and low-confidence code paths."""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from xgb_stage import XGBStage, UNKNOWN

s = XGBStage("models")
samples = json.loads(Path("tests/samples.json").read_text())

r = s.predict(samples["known"])
assert not r["is_unknown"] and r["attack_family"] in s.classes and r["confidence"] >= s.threshold
print("PASS known family   :", r["attack_family"], r["confidence"])

r = s.predict(samples["unknown"])
assert r["is_unknown"] and r["attack_family"] == UNKNOWN and r["confidence"] < s.threshold
print("PASS unknown/novel  :", r["attack_family"], "closest =", r["closest_family"], r["confidence"])

good = samples["known"]
bad_cases = {
    "missing feature": {k: v for k, v in good.items() if k != "src_bytes"},
    "extra feature":   {**good, "if_score": 0.7},
    "NaN value":       {**good, "src_bytes": float("nan")},
    "inf value":       {**good, "dst_bytes": float("inf")},
    "negative value":  {**good, "src_packets": -1},
    "string value":    {**good, "src_packets": "12"},
    "bad one-hot":     {**good, "proto_udp": 1},
    "not a dict":      [1, 2, 3],
}
for name, x in bad_cases.items():
    try:
        s.predict(x)
        raise SystemExit(f"FAIL: '{name}' was accepted")
    except ValueError:
        print("PASS rejected       :", name)
print("\nAll XGBStage tests passed.")