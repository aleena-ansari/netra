import joblib
import pandas as pd


class IFStage:
    """Isolation Forest inference adapter."""

    def __init__(self, stream, model_dir="models"):
        self.stream = stream

        if stream == "aws_vpc_flow_log":
            filename = "iforest_aws_vpc_flow_log.joblib"
        elif stream == "cisco_asa":
            filename = "iforest_cisco_asa.joblib"
        else:
            raise ValueError(f"Unknown stream: {stream}")

        self.model = joblib.load(f"{model_dir}/{filename}")
        self.features = list(self.model.feature_names_in_)

        self.version = f"{stream}-if-v1"

    def score(self, event):
        missing = [f for f in self.features if f not in event]

        if missing:
            raise ValueError(
                f"Missing IF features for {self.stream}: {missing}"
            )

        X = pd.DataFrame(
            [[event[f] for f in self.features]],
            columns=self.features,
        ).astype("float32")

        raw_score = float(self.model.score_samples(X)[0])
        decision = float(self.model.decision_function(X)[0])

        # Isolation Forest: decision_function < 0 => anomaly
        is_anomaly = decision < 0.0

        return {
            "anomaly_score": round(-raw_score, 6),
            "decision_score": round(decision, 6),
            "is_anomaly": bool(is_anomaly),
            "model_version": self.version,
        }