from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime
import uuid

@dataclass
class SeedAnalysis:
    analysis_id: str
    sample_guid: str
    sample_id: str
    predicted_class: str
    probability: float
    probability_vector: list
    features: Dict[str, Any]
    model_version: str
    created_at: datetime

    @staticmethod
    def create(predicted_class: str, probability: float, probability_vector: list, features: Dict[str, Any], model_version: str = "seed_cnn_v1", sample_guid: str | None = None):
        now = datetime.utcnow()
        return SeedAnalysis(
            analysis_id=str(uuid.uuid4()),
            sample_guid= sample_guid or str(uuid.uuid4()),
            sample_id=str(uuid.uuid4()),
            predicted_class=predicted_class,
            probability=float(probability),
            probability_vector=list(probability_vector) if probability_vector is not None else [],
            features=features or {},
            model_version=model_version,
            created_at=now
        )
