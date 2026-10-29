"""Native lm-evaluation-harness adapter for the custom JAX pilot consumer."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from lm_eval.api.model import LM

from consumer import LocalCausalModel, load_package
import model


class C02PilotLM(LM):
    """Greedy local model adapter with capped requests and candidate-owned math."""

    def __init__(self, package_dir: str | Path):
        super().__init__()
        self.loaded: LocalCausalModel = load_package(package_dir)
        self.max_length = self.loaded.config.context
        self.tokenizer = self.loaded.tokenizer

    @property
    def tokenizer_name(self):
        return f"c02-local-{self.loaded.manifest['branch']}"

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _context_and_target_ids(self, context: str, continuation: str):
        """Score joined-text tokens whose tokenizer offsets touch the continuation.

        Tokenizers reports offsets in Unicode character positions, not UTF-8
        byte positions. Joint encoding keeps a BPE merge at the boundary intact.
        """
        joined = context + continuation
        encoded = self.tokenizer.encode(joined, add_special_tokens=False)
        boundary = len(context)
        first_target = next(
            (index for index, (_, end) in enumerate(encoded.offsets) if end > boundary),
            len(encoded.ids),
        )
        return encoded.ids[:first_target], encoded.ids[first_target:]

    def _score(
        self, context_ids: list[int], target_ids: list[int]
    ) -> tuple[float, bool]:
        context = list(context_ids) or [1]
        total = 0.0
        greedy = True
        offset = 0
        while offset < len(target_ids):
            # A single target can be predicted from the final position of a
            # full max_length context. The older max_length-1 slice silently
            # discarded one usable token on the final (or singleton) window.
            context_tail = context[-self.max_length :]
            capacity = self.max_length - len(context_tail) + 1
            count = min(capacity, len(target_ids) - offset)
            targets = target_ids[offset : offset + count]
            inputs = context_tail + targets[:-1]
            positions = jnp.arange(len(inputs), dtype=jnp.int32)[None, :]
            logits = model.logits(
                self.loaded.params,
                jnp.asarray([inputs], dtype=jnp.int32),
                self.loaded.config,
                valid_mask=jnp.ones((1, len(inputs)), dtype=jnp.bool_),
                segment_ids=jnp.ones((1, len(inputs)), dtype=jnp.int32),
                position_ids=positions,
            )[0]
            start = len(context_tail) - 1
            selected = logits[start : start + count].astype(jnp.float32)
            log_probs = jax_log_softmax(selected)
            ids = jnp.asarray(targets, dtype=jnp.int32)
            chosen = jnp.take_along_axis(log_probs, ids[:, None], axis=-1)[:, 0]
            total += float(jnp.sum(chosen))
            greedy = greedy and bool(jnp.all(jnp.argmax(selected, axis=-1) == ids))
            context.extend(targets)
            offset += count
        return total, greedy

    def loglikelihood(self, requests):
        results = []
        for request in requests:
            context, continuation = request.args
            context_ids, target_ids = self._context_and_target_ids(
                context, continuation
            )
            if not target_ids:
                results.append((0.0, True))
            else:
                results.append(self._score(context_ids, target_ids))
        return results

    def loglikelihood_rolling(self, requests):
        return [self._score([], self._ids(request.args[0]))[0] for request in requests]

    def generate_until(self, requests):
        results = []
        for request in requests:
            context, kwargs = request.args
            if kwargs.get("do_sample", False) or kwargs.get("temperature", 0) not in (
                0,
                0.0,
            ):
                raise ValueError("the frozen C02 adapter supports greedy decoding only")
            unsupported = set(kwargs) - {
                "until",
                "max_gen_toks",
                "max_new_tokens",
                "do_sample",
                "temperature",
            }
            if unsupported:
                raise ValueError(
                    f"unsupported generation options: {sorted(unsupported)}"
                )
            maximum = int(kwargs.get("max_gen_toks", kwargs.get("max_new_tokens", 32)))
            generated = self.loaded.generate(context, maximum)
            text = generated["text"]
            stops = kwargs.get("until", [])
            if isinstance(stops, str):
                stops = [stops]
            positions = [text.find(stop) for stop in stops if stop and stop in text]
            if positions:
                text = text[: min(positions)]
            results.append(text)
        return results


def jax_log_softmax(logits):
    import jax

    return jax.nn.log_softmax(logits, axis=-1)
