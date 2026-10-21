"""Configuration contracts for the SciFact encoder lifecycle pilot."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int = 512
    max_length: int = 96
    width: int = 64
    layers: int = 2
    heads: int = 4
    mlp_width: int = 128
    dropout: float = 0.1
    tied_mlm_head: bool = True
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    unk_id: int = 1
    mask_id: int = 4

    def validate(self):
        if self.vocab_size < 8 or self.max_length < 4 or self.width < 4:
            raise ValueError("vocabulary, context, and width must be positive")
        if self.layers < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("layers and heads must be positive and divide width")
        if self.mlp_width < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid MLP width or dropout")
        special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id, self.mask_id}
        if len(special) != 5 or min(special) < 0 or max(special) >= self.vocab_size:
            raise ValueError("special token IDs must be unique and in vocabulary")

    def as_dict(self):
        self.validate()
        return asdict(self)

    def fingerprint(self):
        value = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    warmup_steps: int = 1
    total_steps: int = 6
    seed: int = 20261021
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def validate(self):
        if self.batch_size < 1 or self.total_steps < 1 or self.warmup_steps < 0:
            raise ValueError("invalid batch or step count")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer rate or decay")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1 or self.epsilon <= 0:
            raise ValueError("invalid Adam parameters")

    def as_dict(self):
        self.validate()
        return asdict(self)

    def fingerprint(self):
        value = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


TARGET_CONFIG = EncoderConfig(
    vocab_size=49152,
    max_length=512,
    width=1024,
    layers=24,
    heads=16,
    mlp_width=3072,
    dropout=0.1,
    tied_mlm_head=True,
)
