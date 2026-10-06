"""Module 1, Stage 2b: XGBoost attack-family classifier.

Thin adapter around the saved Stage-3 artifact (stage3_xgb_open_set.joblib).
It does NOT retrain or modify the model. It adds: strict input validation, the
mandatory feature order, label decoding, and the UNKNOWN/NOVEL rule.
The model has no NORMAL class and must only be called on events the
Isolation Forest flagged as anomalous.
"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Order is mandatory: the saved model stores no feature names.
FEATURES = [
    "src_packets", "dst_packets", "total_packets",
    "src_bytes", "dst_bytes", "total_bytes",
    "src_mean_pkt_size", "dst_mean_pkt_size",
    "proto_tcp", "proto_udp", "proto_icmp", "proto_other",
]
UNKNOWN = "UNKNOWN/NOVEL"


def build_xgb_features(src_packets, dst_packets, src_bytes, dst_bytes, proto):
    """Build the 12-feature dict from per-direction flow counts.

    CAVEAT: mean packet size here is bytes / packets (as in the technical PDF).
    The CIC part of her training code used the dataset's own 'Fwd Pkt Len Mean'
    column, so confirm her definition before trusting scores on new data.
    proto: 'tcp' | 'udp' | 'icmp' | anything else -> proto_other.
    """
    p = str(proto).lower()
    return {
        "src_packets": src_packets, "dst_packets": dst_packets,
        "total_packets": src_packets + dst_packets,
        "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "total_bytes": src_bytes + dst_bytes,
        "src_mean_pkt_size": src_bytes / src_packets if src_packets > 0 else 0.0,
        "dst_mean_pkt_size": dst_bytes / dst_packets if dst_packets > 0 else 0.0,
        "proto_tcp": int(p == "tcp"), "proto_udp": int(p == "udp"),
        "proto_icmp": int(p == "icmp"),
        "proto_other": int(p not in ("tcp", "udp", "icmp")),
    }


class XGBStage:
    def __init__(self, model_dir="models"):
        d = Path(model_dir)
        art = joblib.load(d / "stage3_xgb_open_set.joblib")
        self.model = art["model"]
        self.threshold = float(art["threshold"])
        if self.model.n_features_in_ != len(FEATURES):
            raise ValueError(f"model expects {self.model.n_features_in_} features, "
                             f"adapter defines {len(FEATURES)}")
        meta = d / "xgb_meta_v1.json"
        if meta.exists():                       # preferred: no pickle-version risk
            self.classes = json.loads(meta.read_text())["classes"]
        else:
            self.classes = [str(c) for c in joblib.load(d / "stage3_label_encoder.pkl").classes_]

    def _validate(self, feats):
        if not isinstance(feats, dict):
            raise ValueError("features must be a dict")
        missing = [f for f in FEATURES if f not in feats]
        extra = [k for k in feats if k not in FEATURES]
        if missing or extra:
            raise ValueError(f"feature mismatch. missing={missing} unexpected={extra}")
        row = []
        for f in FEATURES:
            v = feats[f]
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise ValueError(f"{f} must be numeric, got {type(v).__name__}")
            if not math.isfinite(v):
                raise ValueError(f"{f} is not finite: {v}")
            if v < 0:
                raise ValueError(f"{f} is negative: {v}")
            row.append(float(v))
        if sum(feats[f] for f in FEATURES[8:]) != 1:
            raise ValueError("protocol one-hot must have exactly one 1")
        return row

    def predict(self, feats):
        """feats: dict with the 12 features. Returns attack_family, confidence, is_unknown."""
        row = self._validate(feats)
        X = pd.DataFrame([row], columns=FEATURES).astype("float32")
        proba = self.model.predict_proba(X)[0]
        k = int(proba.argmax())
        conf = float(proba[k])
        unknown = conf < self.threshold
        top3 = np.argsort(proba)[::-1][:3]
        return {
            "attack_family": UNKNOWN if unknown else self.classes[k],
            "closest_family": self.classes[k],       # useful for the analyst even if unknown
            "confidence": round(conf, 4),
            "is_unknown": unknown,
            "threshold": self.threshold,
            "top3": [{"family": self.classes[i], "p": round(float(proba[i]), 4)} for i in top3],
        }