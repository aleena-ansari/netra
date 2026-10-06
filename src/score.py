"""
Module 1 entry point.

Pipeline:
    backend rule hit
        -> immediate attack result

    otherwise
        -> Isolation Forest anomaly gate
        -> if below threshold: benign
        -> if above threshold: XGBoost family classifier
        -> JSON-compatible result

No labels are used by the Isolation Forest stage.

The Isolation Forest stage must receive the SAME feature representation
used during its training. This module therefore treats IF scoring as an
adapter/stage rather than passing a raw event dict directly to sklearn.
"""

import math
import time


# XGBoost stage
from .xgb_stage import XGBStage, build_xgb_features, UNKNOWN


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEVERITY_BY_FAMILY = {
    "Backdoor / Persistence": "critical",
    "Data Exfiltration": "critical",
    "Shellcode / Payload Execution": "high",
    "Exploitation / RCE": "high",
    "Credential Attack / Brute Force": "high",
    "DoS / Flooding": "medium",
    "Reconnaissance / Scanning": "medium",
    "Fuzzing": "low",
}


# These are the fields required by the public score_event() contract.
REQUIRED = [
    "event_id",
    "ts",
    "src_packets",
    "dst_packets",
    "src_bytes",
    "dst_bytes",
    "proto",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _finite_nonnegative(value, name):
    """Validate a numeric event field."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a finite number >= 0, got {value!r}"
        )


def _clamp01(value):
    """Clamp a numeric score into [0, 1]."""
    return min(max(float(value), 0.0), 1.0)


# ---------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------

class Detector:
    """
    High-level detector.

    if_stage:
        Adapter around the trained Isolation Forest.
        It must expose:

            score(event) -> float

        where the adapter is responsible for turning the event into the
        exact IF feature representation used during training.

    xgb_stage:
        XGBStage instance.
    """

    def __init__(self, if_stage, xgb_stage=None, model_dir="models"):
        self.if_stage = if_stage

        self.xgb = xgb_stage or XGBStage(model_dir)

        if not hasattr(self.if_stage, "score"):
            raise TypeError(
                "if_stage must provide score(event) -> float"
            )

        # Keep the version visible in every result.
        if_version = getattr(self.if_stage, "version", "if_unknown")
        xgb_version = getattr(self.xgb, "version", "xgb_unknown")

        self.version = f"m1-v1|if:{if_version}|xgb:{xgb_version}"

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------

    def score_event(self, event, rule_hit=None):
        """
        Score one event.

        Order is deliberately:

            1. validate event
            2. backend rule
            3. Isolation Forest
            4. threshold gate
            5. XGBoost family classification
        """

        t0 = time.perf_counter()

        self._validate_event(event)

        out = {
            "event_id": event["event_id"],
            "ts": event["ts"],
            "entity": event.get("src_ip", ""),

            "model_version": self.version,

            "detected_by": "none",

            "attack_family": None,
            "closest_family": None,
            "family_confidence": None,
            "top3": None,

            "if_score": None,
            "score": 0.0,

            "label": "benign",
            "severity": "low",
            "is_attack": False,

            "severity_source": None,
            "explanation": None,

            "latency_ms": None,
        }

        # =============================================================
        # Stage 1: backend rule
        # =============================================================

        if rule_hit:
            category = rule_hit.get("category")
            severity = rule_hit.get(
                "severity",
                SEVERITY_BY_FAMILY.get(category, "high"),
            )

            out.update(
                detected_by="rule",
                is_attack=True,
                score=1.0,
                label="attack",
                severity=severity,
                attack_family=category,
                explanation=(
                    f"Matched backend rule "
                    f"{rule_hit.get('rule_id', '?')}"
                ),
                severity_source="rule",
            )

            out["latency_ms"] = self._latency_ms(t0)
            return out

        # =============================================================
        # Stage 2a: Isolation Forest anomaly gate
        # =============================================================

        if_result = self.if_stage.score(event)

        # Support either:
        #
        #   score(event) -> float
        #
        # or:
        #
        #   score(event) -> {"score": ..., "threshold": ...}
        #
        if isinstance(if_result, dict):
            if_score = float(if_result["score"])
            threshold = float(if_result["threshold"])
        else:
            if_score = float(if_result)
            threshold = float(
                getattr(self.if_stage, "threshold", 0.0)
            )

        if_score = _clamp01(if_score)

        out["if_score"] = round(if_score, 4)

        # -------------------------------------------------------------
        # Below anomaly threshold
        # -------------------------------------------------------------

        if if_score < threshold:
            out.update(
                detected_by="anomaly_model",
                is_attack=False,
                score=round(if_score, 4),
                label="benign",
                severity="low",
                severity_source="anomaly_threshold",
                explanation=(
                    f"Anomaly score {if_score:.2f} below "
                    f"threshold {threshold:.2f}"
                ),
            )

            out["latency_ms"] = self._latency_ms(t0)
            return out

        # =============================================================
        # Stage 2b: XGBoost family classification
        # =============================================================

        feats = build_xgb_features(
            event["src_packets"],
            event["dst_packets"],
            event["src_bytes"],
            event["dst_bytes"],
            event["proto"],
        )

        result = self.xgb.predict(feats)

        attack_family = result.get("attack_family")
        closest_family = result.get("closest_family")
        confidence = result.get("confidence")
        top3 = result.get("top3")
        is_unknown = bool(result.get("is_unknown", False))

        if confidence is not None:
            confidence = float(confidence)

        # -------------------------------------------------------------
        # Unknown / low-confidence XGBoost classification
        # -------------------------------------------------------------

        if is_unknown:
            out.update(
                detected_by="anomaly_model",
                is_attack=True,
                score=round(if_score, 4),
                label="suspicious",
                severity="medium",
                attack_family=UNKNOWN,
                closest_family=closest_family,
                family_confidence=confidence,
                top3=top3,
                severity_source="anomaly_model",
                explanation=(
                    f"Anomalous traffic detected "
                    f"(score {if_score:.2f}), but the family "
                    f"classifier is uncertain"
                ),
            )

        # -------------------------------------------------------------
        # Known XGBoost family
        # -------------------------------------------------------------

        else:
            severity = SEVERITY_BY_FAMILY.get(
                attack_family,
                "medium",
            )

            out.update(
                detected_by="anomaly_model",
                is_attack=True,
                score=round(if_score, 4),
                label="attack",
                severity=severity,
                attack_family=attack_family,
                closest_family=closest_family,
                family_confidence=confidence,
                top3=top3,
                severity_source="provisional_family_table",
                explanation=(
                    f"Anomalous traffic detected "
                    f"(score {if_score:.2f}) and classified as "
                    f"{attack_family} "
                    f"with {confidence:.0%} confidence"
                    if confidence is not None
                    else
                    f"Anomalous traffic detected "
                    f"(score {if_score:.2f}) and classified as "
                    f"{attack_family}"
                ),
            )

        out["latency_ms"] = self._latency_ms(t0)

        return out

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_event(event):
        if not isinstance(event, dict):
            raise ValueError("event must be a dict")

        missing = [
            key for key in REQUIRED
            if key not in event
        ]

        if missing:
            raise ValueError(
                f"event missing fields: {missing}"
            )

        for key in (
            "src_packets",
            "dst_packets",
            "src_bytes",
            "dst_bytes",
        ):
            _finite_nonnegative(event[key], key)

    # -----------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------

    @staticmethod
    def _latency_ms(t0):
        return round(
            (time.perf_counter() - t0) * 1000,
            3,
        )


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------

def score_event(event, rule_hit=None, if_stage=None, xgb_stage=None):
    """
    Functional entry point.

    Example:

        result = score_event(
            event,
            rule_hit=rule_hit,
            if_stage=if_stage,
            xgb_stage=xgb_stage,
        )
    """

    if if_stage is None:
        raise ValueError(
            "if_stage is required; pass the trained "
            "Isolation Forest adapter"
        )

    detector = Detector(
        if_stage=if_stage,
        xgb_stage=xgb_stage,
    )

    return detector.score_event(
        event,
        rule_hit=rule_hit,
    )