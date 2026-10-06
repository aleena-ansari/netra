from .if_stage import IFStage
from .xgb_stage import XGBStage


class ATDEPipeline:
    """
    Final ATDE inference pipeline.

    Flow:
        IF -> XGB

    XGB is only evaluated when Isolation Forest
    flags the event as anomalous.
    """

    def __init__(self, stream, model_dir="models"):
        self.stream = stream
        self.if_stage = IFStage(
            stream=stream,
            model_dir=model_dir,
        )
        self.xgb_stage = XGBStage(
            model_dir=model_dir,
        )

    def predict(self, if_features, xgb_features):
        # Stage 1: Isolation Forest
        if_result = self.if_stage.score(if_features)

        # Normal traffic stops here
        if not if_result["is_anomaly"]:
            return {
                "stream": self.stream,
                "is_anomaly": False,
                "attack_family": None,
                "confidence": 0.0,
                "is_unknown": False,
                "if_score": if_result["anomaly_score"],
                "if_decision": if_result["decision_score"],
                "xgb": None,
            }

        # Stage 2: XGBoost
        xgb_result = self.xgb_stage.predict(xgb_features)

        return {
            "stream": self.stream,
            "is_anomaly": True,
            "attack_family": xgb_result["attack_family"],
            "confidence": xgb_result["confidence"],
            "is_unknown": xgb_result["is_unknown"],
            "if_score": if_result["anomaly_score"],
            "if_decision": if_result["decision_score"],
            "xgb": xgb_result,
        }


def create_pipeline(stream, model_dir="models"):
    return ATDEPipeline(
        stream=stream,
        model_dir=model_dir,
    )


if __name__ == "__main__":
    print("ATDE pipeline module loaded successfully.")

    for stream in ["aws_vpc_flow_log", "cisco_asa"]:
        pipeline = ATDEPipeline(stream)

        print(f"{stream}:")
        print(f"  IF features: {len(pipeline.if_stage.features)}")
        print(f"  XGB features: {pipeline.xgb_stage.model.n_features_in_}")