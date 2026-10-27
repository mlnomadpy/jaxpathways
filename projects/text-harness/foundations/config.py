"""Serializable model and CPU-pilot configuration."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 384
    context: int = 32
    width: int = 32
    layers: int = 2
    heads: int = 4
    mlp_width: int = 64
    dropout: float = 0.1
    tied_embeddings: bool = True

    def validate(self):
        if self.vocab_size < 260 or self.context < 4 or self.width < 4:
            raise ValueError(
                "vocabulary, context, or width is below the supported minimum"
            )
        if self.layers < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError(
                "layers and heads must be positive and heads must divide width"
            )
        if self.mlp_width < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid MLP width or dropout rate")

    def as_dict(self):
        self.validate()
        return asdict(self)

    def fingerprint(self):
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4
    learning_rate: float = 0.001
    min_learning_rate: float = 0.0002
    warmup_steps: int = 2
    total_steps: int = 20
    seed: int = 20261020
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def validate(self):
        if self.batch_size < 1 or self.total_steps < 1 or self.warmup_steps < 0:
            raise ValueError("invalid training step or batch count")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError(
                "learning rates must be nonnegative and peak rate positive"
            )
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1 or self.epsilon <= 0:
            raise ValueError("invalid Adam parameters")

    def as_dict(self):
        self.validate()
        return asdict(self)

    def fingerprint(self):
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
