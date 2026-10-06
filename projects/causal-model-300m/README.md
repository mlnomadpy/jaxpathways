# Capstone: train and release a 300M causal language model

**Status: planned specification.** Inherits the [shared lifecycle gates](../../docs/model-lifecycle-capstones.md) and [tooling guide](../../docs/model-lifecycle-tooling.md). No large-model training or publication has been executed. This is a separate generative project from the MLM/embedding model.

Outcome: build a small causal model from a versioned corpus, adapt it for a declared use case, compare full fine-tuning with LoRA, evaluate its limitations, and operate a reproducible generation service. Production readiness applies to that use case; a 300M model is not presumed to be a broadly capable assistant.

## Starting point and architecture

Extend the [text harness](../text-harness/README.md), retaining its actual causal checks, full-state recovery, prefill/decode cache, precision and export contracts. Use TEXT-1..3/5, ADAPT, DATA/EVAL, BRIDGE/NUM and SERVE/OPS/SCALE preparation. PREF is needed only for the optional preference stage.

A concrete first architecture hypothesis uses 24 blocks, hidden width 1,024, 16 full attention heads, a two-matrix GELU MLP of width 3,072, vocabulary 49,152, affine pre-LayerNorm/final norm, bias-free linear layers, RoPE and tied input/output embeddings. This has the same approximately 302M unique-parameter budget as the embedding project's encoder candidate, with a causal attention mask and next-token output head rather than an MLM transform. Verify the actual parameter tree. Start with a declared 2,048-token context after tiny/pilot tests; longer context is another training/evaluation claim. Compare GQA or a gated MLP later with recalculated counts and cache layouts.

## Data and input recipe

Begin with an English general/domain text task. A pilot mixture hypothesis is 60% curated general text, 25% relevant domain/reference material and 15% licensed dialogue/instruction-like prose, measured in consumed non-padding tokens. These are recipe categories, not approved datasets or optimal percentages. Keep code/math or multilingual content as explicitly weighted variants when the target requires them.

Pin sources, audit rights/provenance, deduplicate and split by source/document before chunking. Compare tokenizer length/coverage/cost on training data, freeze special IDs and normalization, and check benchmark overlap at every stage. Preserve EOS and document boundaries; packed tokens must not create accidental targets across unrelated documents. Record chosen attention isolation versus cross-document context policies.

## Training phases and handoffs

| Phase | Objective and learner work | Artifact / promotion evidence |
| --- | --- | --- |
| 0: tiny correctness and recipe pilot | Independent next-token loss, causal masks, tokenizer round trip, global token weighting and cache parity; measure real-data throughput/memory | Frozen pilot protocol and actual parameter/runtime report |
| 1: base pretraining | Train from scratch on next-token cross-entropy with a valid-target mask, stable precision and recorded token/schedule clocks | Complete base checkpoint; held-out NLL by source/length, actual generation failures and fresh-process replay |
| 2: domain continuation | Continue next-token learning on a declared domain/general mixture; compare against skipping this phase under a fair resource budget | Continued checkpoint and target-domain gain versus general retention; optimizer reset/continuation decision recorded |
| 3: supervised instruction adaptation | Response-only SFT with a fixed chat template; compare full fine-tuning and real Transformer LoRA | Adapted model/adapter, correct shifted role masks, held-out behavior/retention and merged/unmerged parity |
| 4: optional preference adaptation | Sequence DPO on curated preferences, or learned reward/value/sequence PPO after its dedicated lessons | Separate preference checkpoint with label provenance, response-length/mask checks, external quality and drift; no synthetic-label human-alignment claim |
| 5: release qualification | Evaluate loaded exports, precision/cache variants, generation API and immutable Hub package | Numerical/task/runtime gates and a reproducible consumer/service report |

The base objective is:

\[
L=-\frac{\sum_{b,t}m_{b,t}\log p_\theta(x_{b,t+1}\mid x_{b,\le t})}{\sum_{b,t}m_{b,t}}.
\]

Define \(m\) on target positions; PAD and disallowed packed-boundary targets contribute neither loss nor count. SFT changes this mask to the chosen response targets while preserving prompt context. Check one shifted sequence by hand and inject an off-by-one mask error. Ordinary accumulation must reproduce a token-weighted full-batch update under controlled randomness.

Each phase records parent checkpoint, data/tokenizer/template, precision, objective and schedule identity. Exact resume is tested within phases; a deliberately changed objective is a new experiment. From-scratch, continued-pretraining and imported-base variants must not share misleading training-history claims.

## Evaluation and monitoring

Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) through a verified native adapter or a numerically qualified supported conversion. Pin exact task versions, prompts, few-shot examples, scoring conventions and decoding. Start with an attainable subset of completion/common-sense/task-domain tests, then expand only for claimed capabilities. Include a pinned small pretrained baseline and chance/simple baselines where applicable. MTEB is not the primary benchmark for generation.

Report held-out token NLL/perplexity under the **same tokenizer**, domain task scores, fixed generation errors, instruction adherence, stop-token/length behavior, repetitions and unsupported answers. Code-generation benchmarks require an isolated test executor if that capability is included. Final test sets do not tune sampling or templates. A weak score is reported, not hidden behind cherry-picked continuations.

Track valid tokens per source, loss/gradients/updates, learning rate, clipping/nonfinite state, padding, memory and data/compute wait in MLflow or W&B. At adaptation stages add response-token coverage, base retention and preference/length statistics. Fixed generation examples use saved prompts/seeds/decoding; changing temperature can change apparent quality without improving weights.

Required figures: source exposure and token-length distributions, train/validation NLL by token budget, base→continued→SFT task/retention comparisons, actual failed generations, layer/cache precision error, and time-to-first-token/inter-token/end-to-end latency under load. Explain each plot's measurement boundary.

## Serving and Hugging Face release

Save tokenizer, architecture, tied weights, generation configuration, chat template where applicable and the tested loader. Verify full-forward versus incremental cache logits, changed batch/prompt lengths, EOS, capacity overflow and cache isolation between clients. Qualify prefill/decode batching, cancellation, bounded queues and resource limits. Record cache precision separately from weights/activations/accumulation.

Follow the [shared publication workflow](../embedding-model-300m/RELEASE.md), substituting the causal consumer contract. A Transformers-compatible release requires an implemented mapping; a custom JAX decoder is not automatically supported by vLLM or `AutoModelForCausalLM`. Test every advertised path after an immutable cold download. Rollback restores model/tokenizer/template/decoder together.

## Learner review

Retain all stage manifests/checkpoints, one data/objective ablation, a recovery failure/repair, independent mask/loss checks, benchmark inventory including failures, actual W&B/MLflow artifacts, generation/service evidence and release canaries. A reviewer should be able to explain why each phase exists and reproduce the selected model's behavior without the training workspace.
