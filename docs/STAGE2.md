# Stage 2 Roadmap for `aumlit reshell`

## High-Leverage Upgrades
1. **Geometry DSL** – normalize tensor shapes and arithmetic into a tiny shape algebra; regex becomes a feeder, planner consumes emitted constraints (e.g., `d == 2048`, `C ∈ {4,8}`, `scale | 8`).
2. **Greedy planner** – score probes by cardinality collapse (`log|H_before| - log|H_after_if_fail|`) and respect tier ordering (EMB→HEAD→LATENT→VISION→LLM) with timeout-aware engine selection.
3. **Enriched cache** – hash key plus outcome class, extracted ints, engine id, failing op, timings, and cross-artifact reuse for shared headers.
4. **Minimum validation** – add stability, precision, and I/O sanity checks; flag unstable engines, dtype issues, or degenerate outputs.
5. **Obligations with provenance** – attach origin info to each requirement (source constraint, op kind, proof line).
6. **ONNX and GGUF signals** – leverage shape inference, dynamic axis checks, custom op detection, and GGUF header data before expensive probes.
7. **xFormers layout oracle** – use mem-efficient toggle to infer attention layout and feed back into planner.
8. **Error corpus & golden tests** – maintain `rules/error_rules.yaml` with exemplars and golden tests to ensure classifier resilience.
9. **CLI ergonomics** – richer flags (`--fmt`, `--guess`, presets `--fast`/`--deep`) and proof limits.
10. **ComfyUI bridge** – export minimal graphs with TODO nodes, shape annotations, and re-prove button inside Comfy.

## Failure Modes to Pin Down Early
- Silent success with wrong semantics.
- Error phrasing drift.
- Probe side-effects from JIT/caching.
- OOM from pathological headers.
- Heisenbugs from parallelism.

## Quick Hits
- Implement constraint objects and switch classifier to emit `ConstraintSet`.
- Add latent scales 4 & 32 with single conv probes.
- Record failing op kind uniformly.
- Support `--proof-limit` and emit best partial trace.
- Ship `examples/` with TorchScript, ONNX, GGUF fixtures.

## Stretch Ideas
- Witness minimisation for smallest validating inputs.
- ONNX graph fingerprinting to hint model family.
- Constraint SAT solver to prune candidate space.

## Metrics
- **Collapse rate** – probes to reach ready state.
- **Correctness** – accuracy of inferred dimensions and types.
- **Robustness** – classifier coverage across engine versions.
- **Safety** – watchdog kills, resource usage stats.
- **UX** – time-to-first-useful-obligation.

