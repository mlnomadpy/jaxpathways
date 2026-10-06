# Systems lesson execution evidence

All 83 canonical lesson scripts and all 83 notebook companions passed the final full CPU smoke after the recovery review fix. The machine-readable receipt is curriculum/validation.json; generated notebooks retain source-matched executed outputs and figures.

- welcome-03 executes a completed prediction, compares it with an independent NumPy result and checks the requested backend. The TPU command explicitly selects TPU and fails when unavailable; CPU is not accepted as a fallback.
- distributed-03 performs actual collectives on four logical CPU devices, compares gradients with NumPy, checks local shapes and lowered collectives, and separates a modeled communication budget from synchronized local measurements.
- distributed-04 saves and restores complete momentum, random and sampler state using Orbax. Its resume mode reopens the accepted step-one fixture rather than overwriting it. A second Python process reproduced the same output while all twelve checkpoint-file hashes remained unchanged. Independent review repeated that experiment.
- recovery-05 covers asynchronous checkpoint acceptance and failure; performance-03 through performance-05 retain actual profiler/lowering evidence and measured derivative checks while identifying modeled memory or roofline quantities.

The logical-device exercises do real JAX computation. They do not establish physical TPU execution, multi-host fault tolerance or accelerator performance. Executable target paths remain separate qualification work. No paid infrastructure or external device was provisioned.
