"""C02 decoder and training configuration, adapted from the accepted L05 base."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 512
    context: int = 512
    width: int = 32
    layers: int = 2
    heads: int = 4
    mlp_width: int = 64
    dropout: float = 0.1
    tied_embeddings: bool = True
    rope_base: float = 10000.0

    def validate(self):
        if self.vocab_size < 4 or self.context < 4 or self.width < 4:
            raise ValueError("vocabulary, context, or width is below the minimum")
        if self.layers < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("layers and heads must be positive and divide width")
        head_width = self.width // self.heads
        if head_width % 2:
            raise ValueError("RoPE requires an even per-head width")
        if self.mlp_width < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid MLP width or dropout rate")
        if self.rope_base <= 1:
            raise ValueError("RoPE base must be greater than one")
        if not self.tied_embeddings:
            raise ValueError("this C02 reference currently requires tied embeddings")

    def as_dict(self):
        self.validate()
        return asdict(self)

    def fingerprint(self):
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4
    learning_rate: float = 0.0003
    min_learning_rate: float = 0.00003
    warmup_steps: int = 3
    total_steps: int = 32
    seed: int = 20261021
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    def validate(self):
        if self.batch_size < 1 or self.total_steps < 1 or self.warmup_steps < 0:
            raise ValueError("invalid training step or batch count")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be nonnegative and peak positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1 or self.epsilon <= 0:
            raise ValueError("invalid Adam parameters")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("invalid weight decay or clipping norm")

    def as_dict(self):
        self.validate()
        return asdict(self)

    def fingerprint(self):
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
